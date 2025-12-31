import asyncio
import json
import re
import os
import argparse
import time
from typing import List
from collections import deque
from dotenv import load_dotenv
import google.generativeai as genai

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
    """
    match = re.search(r'Rating:\s*(?:\[\[(\d+)\]\]|(\d+))', response_text)
    if match:
        score_str = match.group(1) if match.group(1) is not None else match.group(2)
        return int(score_str)
    else:
        raise ValueError("Could not extract rating from response: " + response_text)

# Global model & semaphore will be set in main(), then used by async fns
model = None
semaphore: asyncio.Semaphore = None
rate_limiter = None
completed_entries = 0
total_entries_global = 0
progress_lock = None

class RateLimiter:
    """Simple rate limiter for 2000 requests per minute."""
    def __init__(self, max_requests: int = 2000, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()
        self.lock = asyncio.Lock()
        self.total_requests = 0
    
    async def acquire(self):
        """Wait if necessary to respect rate limit, then record request."""
        async with self.lock:
            now = time.time()
            # Remove requests older than the window
            while self.request_times and self.request_times[0] < now - self.window_seconds:
                self.request_times.popleft()
            
            # If we're at the limit, wait until the oldest request expires
            if len(self.request_times) >= self.max_requests:
                wait_time = self.request_times[0] + self.window_seconds - now + 0.1
                if wait_time > 0:
                    print(f"Rate limit reached ({len(self.request_times)}/{self.max_requests}), waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.request_times and self.request_times[0] < now - self.window_seconds:
                        self.request_times.popleft()
            
            # Record this request
            self.request_times.append(time.time())
            self.total_requests += 1
            if self.total_requests % 100 == 0:
                print(f"[Rate Limiter] Total requests made: {self.total_requests}, Current window: {len(self.request_times)}/{self.max_requests}")

# ------------------------------
# LLM evaluators
# ------------------------------
async def evaluate_concept_score(concept: str, sentence_fragment: str, attempts: int = 2) -> int:
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
    for attempt in range(attempts):
        try:
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
                    print(f"      [SLOW] Concept API took {elapsed:.1f}s")
            content = response.text.strip()
            return extract_rating(content)
        except asyncio.TimeoutError:
            print(f"⚠ TIMEOUT: concept scoring attempt {attempt+1} exceeded 60s")
            if attempt == attempts - 1:
                print("Skipping concept score, returning 0.")
                return 0
        except Exception as e:
            error_str = str(e)
            # Check for 429 error and extract retry_delay
            if "429" in error_str or "quota" in error_str.lower():
                retry_delay = 0
                # Try to extract retry_delay from error message
                retry_match = re.search(r'retry.*?(\d+\.?\d*)\s*s', error_str, re.IGNORECASE)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                elif "retry_delay" in error_str:
                    # Try to find seconds value
                    seconds_match = re.search(r'seconds:\s*(\d+)', error_str)
                    if seconds_match:
                        retry_delay = float(seconds_match.group(1))
                if retry_delay > 0:
                    print(f"Warning: Rate limit exceeded, waiting {retry_delay:.1f}s before retry...")
                    await asyncio.sleep(retry_delay)
                    continue
            print(f"Warning: concept scoring attempt {attempt+1} failed: {e}")
            if attempt == attempts - 1:
                print("Skipping concept score, returning 0.")
                return 0

async def evaluate_fluency_score(sentence_fragment: str, attempts: int = 2) -> int:
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
    for attempt in range(attempts):
        try:
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
                    print(f"      [SLOW] Fluency API took {elapsed:.1f}s")
            content = response.text.strip()
            return extract_rating(content)
        except asyncio.TimeoutError:
            print(f"⚠ TIMEOUT: fluency scoring attempt {attempt+1} exceeded 60s")
            if attempt == attempts - 1:
                print("Skipping fluency score, returning 0.")
                return 0
        except Exception as e:
            error_str = str(e)
            # Check for 429 error and extract retry_delay
            if "429" in error_str or "quota" in error_str.lower():
                retry_delay = 0
                # Try to extract retry_delay from error message
                retry_match = re.search(r'retry.*?(\d+\.?\d*)\s*s', error_str, re.IGNORECASE)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                elif "retry_delay" in error_str:
                    # Try to find seconds value
                    seconds_match = re.search(r'seconds:\s*(\d+)', error_str)
                    if seconds_match:
                        retry_delay = float(seconds_match.group(1))
                if retry_delay > 0:
                    print(f"Warning: Rate limit exceeded, waiting {retry_delay:.1f}s before retry...")
                    await asyncio.sleep(retry_delay)
                    continue
            print(f"Warning: fluency scoring attempt {attempt+1} failed: {e}")
            if attempt == attempts - 1:
                print("Skipping fluency score, returning 0.")
                return 0

def harmonic_mean(scores):
    """
    Computes the harmonic mean of the provided scores.
    If any score is zero, the harmonic mean will be zero.
    """
    if any(score == 0 for score in scores):
        return 0
    return len(scores) / sum(1.0 / score for score in scores)

async def llm_judge(sentence: str, concept: str, entry_idx: int, sent_idx: int, total_sents: int) -> dict:
    print(f"    → [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents}: calling concept API...")
    concept_score = await evaluate_concept_score(concept, sentence)
    print(f"    → [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents}: calling fluency API...")
    fluency_score = await evaluate_fluency_score(sentence)
    final_score = harmonic_mean([concept_score, fluency_score])
    print(f"    ✓ [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents} completed - Concept: {concept_score}, Fluency: {fluency_score}, Final: {final_score:.2f}")
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
    print(f"\n{'='*80}")
    print(f"Processing entry {idx+1}/{total_entries} (K={entry.get('K', 'SAE')}, layer={entry['layer']}, level={level}, h_row={h_row})")
    key = (entry["K"] if not is_diffmean and ("K" in entry) else "SAE", entry["layer"], level, h_row)
    concept_desc = concept_map.get(key)
    if concept_desc is None:
        print(f"⚠ Warning: No concept for {key}")
        sentence_results = []
    else:
        sentences = entry.get("steered_sentences", [])
        total_sents = len(sentences)
        print(f"  Concept: {concept_desc[:100]}..." if len(concept_desc) > 100 else f"  Concept: {concept_desc}")
        print(f"  Evaluating {total_sents} sentences...")
        sentence_results = [
            await llm_judge(sentence, concept_desc, idx+1, s_idx, total_sents)
            for s_idx, sentence in enumerate(sentences)
        ]
        avg_score = sum(r['final_score'] for r in sentence_results) / len(sentence_results) if sentence_results else 0
        print(f"  ✓ Entry {idx+1} complete - Average score: {avg_score:.2f}")
    
    # Update progress counter
    async with progress_lock:
        completed_entries += 1
        if completed_entries % 10 == 0 or completed_entries == total_entries_global:
            progress_pct = (completed_entries / total_entries_global) * 100
            print(f"\n{'*'*80}")
            print(f"PROGRESS: {completed_entries}/{total_entries_global} entries completed ({progress_pct:.1f}%)")
            print(f"{'*'*80}\n")

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
    parser = argparse.ArgumentParser(description="Score steered sentences for concept coverage and fluency.")
    parser.add_argument("--input", required=True, help="Path to steered entries JSON (e.g., causal_output_svd.json)")
    parser.add_argument("--concepts", required=True, help="Path to concepts JSON (e.g., input_descriptions.json)")
    parser.add_argument("--output", required=True, help="Where to write the aggregated results JSON")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use (default: gemini-2.0-flash)")
    parser.add_argument("--ranks", required=True, help='K filter, e.g. "100" or "64,100" or "64-128"')
    parser.add_argument("--layers", required=True, help='Layer filter, e.g. "0,8,16" or "0-16"')
    parser.add_argument("--concurrency", type=int, default=30, help="Max concurrent API calls (default: 30)")
    parser.add_argument(
        "--diffmean",
        action="store_true",
        help="Enable DiffMean baseline"
        )
    parser.add_argument("--api-key-var", default="GEMINI_API_KEY",
                        help="Env var name holding your API key (default: GEMINI_API_KEY)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("LLM JUDGE - SCORING STEERED SENTENCES")
    print("="*80)
    
    # Load .env and get API key
    print("\n[STEP 1/5] Loading configuration...")
    load_dotenv()
    api_key = os.getenv(args.api_key_var)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {args.api_key_var}. "
            f"Create a .env with {args.api_key_var}=sk-... or export it in your shell."
        )
    print(f"  ✓ API key loaded")

    # Initialize global model + semaphore + rate limiter
    print(f"\n[STEP 2/5] Initializing components...")
    global model, semaphore, rate_limiter, completed_entries, total_entries_global, progress_lock
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_limiter = RateLimiter(max_requests=1900, window_seconds=60)  # 5% safety margin
    progress_lock = asyncio.Lock()
    completed_entries = 0
    print(f"  ✓ Model: {args.model}")
    print(f"  ✓ Concurrency: {args.concurrency}")
    print(f"  ✓ Rate limit: 1900 requests/minute")

    # Read inputs
    print(f"\n[STEP 3/5] Loading input files...")
    with open(args.input, "r") as f:
        steered_entries = json.load(f)
    with open(args.concepts, "r") as f:
        concepts = json.load(f)
    print(f"  ✓ Loaded {len(steered_entries)} steered entries from {args.input}")
    print(f"  ✓ Loaded {len(concepts)} concepts from {args.concepts}")

    ranks = parse_int_list(args.ranks)
    layers = parse_int_list(args.layers)
    levels = [i for i in range(len(ranks))] if ranks else [0]

    # Filter entries
    print(f"\n[STEP 4/5] Filtering entries...")
    print(f"  Filters: K in {ranks}, layer in {layers}, level in {levels}")
    filtered = [e for e in steered_entries if ("K" not in e or int(e["K"]) in ranks) and int(e["layer"]) in layers and int(e["hier_level"]) in levels]
    total_entries = len(filtered)
    print(f"  ✓ Selected {total_entries} entries out of {len(steered_entries)}")
    
    # Build concept lookup
    concept_map = {
        (int(c["K"]) if not args.diffmean and "K" in c else "SAE", int(c["layer"]), int(c.get("level", 0)), int(c['h_row'] if 'h_row' in c else c.get('index', 0))): c.get("description", c.get("concept"))
        for c in concepts
        if c.get("description", c.get("concept")) and "TRASH" not in c.get("description", c.get("concept"))
    }
    print(f"  ✓ Built concept map with {len(concept_map)} concepts")

    # Process
    print(f"\n[STEP 5/5] Processing entries and scoring sentences...")
    print(f"{'='*80}")
    total_entries_global = total_entries
    start_time = time.time()
    tasks = [
        asyncio.create_task(process_entry(i, entry, concept_map, total_entries, is_diffmean=args.diffmean))
        for i, entry in enumerate(filtered)
    ]
    all_results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time

    # Save results
    print(f"\n{'='*80}")
    print(f"[SAVING RESULTS]")
    print(f"  Total entries processed: {total_entries}")
    print(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"  Average time per entry: {elapsed_time/total_entries:.2f}s")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved to: {args.output}")
    print(f"\n{'='*80}")
    print("✓ ALL DONE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
