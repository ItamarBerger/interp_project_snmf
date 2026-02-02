import asyncio
import json
import re
import os
import argparse
import time
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
import logging

from experiments.utils import RateLimiter, retry_with_attempts
from utils import setup_logging

logger = logging.getLogger(__name__)

# ------------------------------
# Helpers
# ------------------------------
def parse_int_list(spec: str) -> List[int]:
    """
    Parse "0,2,5-7" into a list of ints: [0,2,5,6,7].
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out

def extract_rating(response_text):
    """
    Extracts the rating from the LLM response.
    Supports both "Rating: [[score]]" and "Rating: score" formats.
    Returns None if extraction fails instead of raising an exception.
    """
    try:
        match = re.search(r'Rating:\s*(?:\[\[(\d+)\]\]|(\d+))', response_text)
        if match:
            score_str = match.group(1) if match.group(1) is not None else match.group(2)
            return int(score_str)
        else:
            logger.info(f"⚠ Warning: Could not extract rating from response: {response_text[:100]}...")
            return None
    except Exception as e:
        logger.info(f"⚠ Warning: Error extracting rating: {e}")
        return None

# Global model & semaphore will be set in main(), then used by async fns
model = None
semaphore: asyncio.Semaphore = None
rate_limiter = None
completed_entries = 0
total_entries_global = 0
progress_lock = None



# ------------------------------
# LLM evaluators
# ------------------------------
async def evaluate_concept_score(concept: str, sentence_fragment: str, attempts: int = 5) -> int:
    """
    Asynchronously evaluates how clearly the specified concept is incorporated in the sentence fragment.
    Returns an integer rating 0-2.
    """
    prompt = f"""Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated.
Point out references to the concept in the fragment, noting any minimal or awkward incorporations.
After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration.
Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence_fragment}
[Sentence Fragment End]"""
    @retry_with_attempts(attempts=attempts, default_value=0)
    async def _inner():
        await rate_limiter.acquire()
        async with semaphore:
            start = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config={"temperature": 0.0}
                ),
                timeout=60.0
            )
            elapsed = time.time() - start
            if elapsed > 10:
                logger.info(f"      [SLOW] Concept API took {elapsed:.1f}s")
        content = response.text.strip()
        rating = extract_rating(content)
        return rating
    return await _inner()


async def evaluate_fluency_score(sentence_fragment: str, attempts: int = 5) -> int:
    """
    Asynchronously evaluates the fluency of the sentence fragment. Returns an integer rating 0-2.
    """
    prompt = f"""Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability.
After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect.
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence_fragment}
[Sentence Fragment End]"""
    @retry_with_attempts(attempts=attempts, default_value=0)
    async def _inner():
        await rate_limiter.acquire()
        async with semaphore:
            start = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config={"temperature": 0.0}
                ),
                timeout=60.0
            )
            elapsed = time.time() - start
            if elapsed > 10:
                logger.info(f"      [SLOW] Fluency API took {elapsed:.1f}s")
        content = response.text.strip()
        rating = extract_rating(content)
        return rating
    return await _inner()


def harmonic_mean(scores):
    """
    Computes the harmonic mean of the provided scores.
    If any score is zero or None, the harmonic mean will be zero.
    """
    # Filter out None values and convert to list
    valid_scores = [s for s in scores if s is not None]
    
    # If any scores were None or if any valid score is 0, return 0
    if len(valid_scores) < len(scores) or any(score == 0 for score in valid_scores):
        return 0
    
    # If no valid scores, return 0
    if not valid_scores:
        return 0
        
    return len(valid_scores) / sum(1.0 / score for score in valid_scores)

async def llm_judge(sentence: str, concept: str, entry_idx: int, sent_idx: int, total_sents: int) -> dict:
    logger.info(f"    → [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents}: calling concept API...")
    concept_score = await evaluate_concept_score(concept, sentence)
    logger.info(f"    → [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents}: calling fluency API...")
    fluency_score = await evaluate_fluency_score(sentence)
    final_score = harmonic_mean([concept_score, fluency_score])
    logger.info(f"    ✓ [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents} completed - Concept: {concept_score}, Fluency: {fluency_score}, Final: {final_score:.2f}")
    return {
        "sentence_index": sent_idx,
        "steered_sentence": sentence,
        "concept_score": concept_score,
        "fluency_score": fluency_score,
        "final_score": final_score,
    }

async def process_entry(idx: int, entry: dict, concept_map: dict, total_entries: int, is_diffmean=False) -> dict:
    """
    Processes one steered entry and returns a single dict that includes
    all scores for its sentences under 'sentence_results'.
    """
    global completed_entries, total_entries_global, progress_lock
    
    level = entry.get('hier_level', 0)
    h_row = entry['h_row'] if 'h_row' in entry else entry.get('index', 0)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing entry {idx+1}/{total_entries} (K={entry.get('K', 'SAE')}, layer={entry['layer']}, level={level}, h_row={h_row})")
    key = (entry["K"] if not is_diffmean and ("K" in entry) else "SAE", entry["layer"], level, h_row)
    concept_desc = concept_map.get(key)
    if concept_desc is None:
        logger.warning(f"⚠ Warning: No concept for {key}")
        sentence_results = []
    else:
        sentences = entry.get("steered_sentences", [])
        total_sents = len(sentences)
        logger.info(f"  Concept: {concept_desc[:100]}..." if len(concept_desc) > 100 else f"  Concept: {concept_desc}")
        logger.info(f"  Evaluating {total_sents} sentences...")
        sentence_results = [
            await llm_judge(sentence, concept_desc, idx+1, s_idx, total_sents)
            for s_idx, sentence in enumerate(sentences)
        ]
        avg_score = sum(r['final_score'] for r in sentence_results) / len(sentence_results) if sentence_results else 0
        logger.info(f"  ✓ Entry {idx+1} complete - Average score: {avg_score:.2f}")
    
    # Update progress counter
    async with progress_lock:
        completed_entries += 1
        if completed_entries % 10 == 0 or completed_entries == total_entries_global:
            progress_pct = (completed_entries / total_entries_global) * 100
            logger.info(f"\n{'*'*80}")
            logger.info(f"PROGRESS: {completed_entries}/{total_entries_global} entries completed ({progress_pct:.1f}%)")
            logger.info(f"{'*'*80}\n")

    return {
        "intervention_sign": entry.get("intervention_sign"),
        "alpha": entry.get("alpha"),
        "kl": entry.get("kl"),
        "K": entry.get("K", "SAE"),
        "layer": entry["layer"],
        "level": level,
        "h_row": h_row,
        "sentence_results": sentence_results,
        "description": concept_desc
    }

# ------------------------------
# Main
# ------------------------------
async def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Score steered sentences for concept coverage and fluency.")
    parser.add_argument("--input", required=True, help="Path to steered entries JSON (e.g., causal_output_svd.json)")
    parser.add_argument("--concepts", required=True, help="Path to concepts JSON (e.g., input_descriptions.json)")
    parser.add_argument("--output", required=True, help="Where to write the aggregated results JSON")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument("--ranks", required=True, help='K filter, e.g. "100" or "64,100" or "64-128"')
    parser.add_argument("--layers", required=True, help='Layer filter, e.g. "0,8,16" or "0-16"')
    parser.add_argument("--concurrency", type=int, default=30, help="Max concurrent API calls (default: 30)")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of entries to process in a batch (default: 20)")
    parser.add_argument(
        "--diffmean",
        action="store_true",
        help="Enable DiffMean baseline"
        )
    parser.add_argument("--api-key-var", default="GEMINI_API_KEY",
                        help="Env var name holding your API key (default: GEMINI_API_KEY)")
    args = parser.parse_args()

    logger.info("Using concurrency:", args.concurrency)
    logger.info("\n" + "="*80)
    logger.info("LLM JUDGE - SCORING STEERED SENTENCES")
    logger.info("="*80)
    
    # Load .env and get API key
    logger.info("\n[STEP 1/5] Loading configuration...")
    load_dotenv()
    api_key = os.getenv(args.api_key_var)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {args.api_key_var}. "
            f"Create a .env with {args.api_key_var}=sk-... or export it in your shell."
        )
    logger.info(f"  ✓ API key loaded")

    # Initialize global model + semaphore + rate limiter
    logger.info(f"\n[STEP 2/5] Initializing components...")
    global model, semaphore, rate_limiter, completed_entries, total_entries_global, progress_lock
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_limiter = RateLimiter(max_requests=1900, window_seconds=60)  # 5% safety margin
    progress_lock = asyncio.Lock()
    completed_entries = 0
    logger.info(f"  ✓ Model: {args.model}")
    logger.info(f"  ✓ Concurrency: {args.concurrency}")
    logger.info(f"  ✓ Rate limit: 1900 requests/minute")

    # Read inputs
    logger.info(f"\n[STEP 3/5] Loading input files...")
    with open(args.input, "r") as f:
        steered_entries = json.load(f)
    with open(args.concepts, "r") as f:
        concepts = json.load(f)
    logger.info(f"  ✓ Loaded {len(steered_entries)} steered entries from {args.input}")
    logger.info(f"  ✓ Loaded {len(concepts)} concepts from {args.concepts}")

    ranks = parse_int_list(args.ranks)
    layers = parse_int_list(args.layers)
    levels = [i for i in range(len(ranks))] if ranks else [0]

    # Filter entries
    logger.info(f"\n[STEP 4/5] Filtering entries...")
    logger.info(f"  Filters: K in {ranks}, layer in {layers}, level in {levels}")
    filtered = [e for e in steered_entries if ("K" not in e or int(e["K"]) in ranks) and int(e["layer"]) in layers and int(e["hier_level"]) in levels]
    total_entries = len(filtered)
    logger.info(f"  ✓ Selected {total_entries} entries out of {len(steered_entries)}")
    
    # Build concept lookup
    concept_map = {
        (int(c["K"]) if not args.diffmean and "K" in c else "SAE", int(c["layer"]), int(c.get("level", 0)), int(c['h_row'] if 'h_row' in c else c.get('index', 0))): c.get("description", c.get("concept"))
        for c in concepts
        if c.get("description", c.get("concept")) and "TRASH" not in c.get("description", c.get("concept"))
    }
    logger.info(f"  ✓ Built concept map with {len(concept_map)} concepts")

    # Process in batches to prevent resource exhaustion
    logger.info(f"\n[STEP 5/5] Processing entries and scoring sentences...")
    logger.info(f"{'='*80}")
    total_entries_global = total_entries
    start_time = time.time()
    
    batch_size = args.batch_size
    all_results = []
    
    for batch_start in range(0, total_entries, batch_size):
        batch_end = min(batch_start + batch_size, total_entries)
        batch = filtered[batch_start:batch_end]
        logger.info(f"\n{'*'*80}")
        logger.info(f"[BATCH {batch_start//batch_size + 1}/{(total_entries + batch_size - 1)//batch_size}] Processing entries {batch_start+1} to {batch_end}...")
        logger.info(f"{'*'*80}")
        
        tasks = [
            asyncio.create_task(process_entry(batch_start + i, entry, concept_map, total_entries, is_diffmean=args.diffmean))
            for i, entry in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        batch_time = time.time() - start_time
        avg_per_entry = batch_time / len(all_results) if all_results else 0
        remaining = total_entries - len(all_results)
        eta_seconds = remaining * avg_per_entry if avg_per_entry > 0 else 0
        logger.info(f"\n{'*'*80}")
        logger.info(f"[BATCH COMPLETE] {len(all_results)}/{total_entries} total entries done. ETA: {eta_seconds/60:.1f} min")
        logger.info(f"{'*'*80}\n")
    
    elapsed_time = time.time() - start_time

    # Save results
    logger.info(f"\n{'='*80}")
    logger.info(f"[SAVING RESULTS]")
    logger.info(f"  Total entries processed: {total_entries}")
    logger.info(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    logger.info(f"  Average time per entry: {elapsed_time/total_entries:.2f}s")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  ✓ Results saved to: {args.output}")
    logger.info(f"\n{'='*80}")
    logger.info("✓ ALL DONE!")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
