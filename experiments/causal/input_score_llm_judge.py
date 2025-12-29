import asyncio
import json
import re
import os
import argparse
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt

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

# Global model, semaphore, and retries will be set in main(), then used by async fns
model = None
semaphore: asyncio.Semaphore = None
retries: int = 5

# ------------------------------
# LLM evaluators
# ------------------------------
def make_evaluate_concept_score(retries: int, gemini_model, max_tokens: int, semaphore: asyncio.Semaphore):
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(retries))
    async def _inner(concept: str, sentence_fragment: str) -> int:
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
        try:
            async with semaphore:
                response = await asyncio.to_thread(
                    gemini_model.generate_content,
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": 0.0}
                )
            return extract_rating(response.text.strip())
        except Exception as e:
            print(f"Warning: concept scoring failed: {e}")
            return 0
    return _inner

def make_evaluate_fluency_score(retries: int, gemini_model, max_tokens: int, semaphore: asyncio.Semaphore):
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(retries))
    async def _inner(sentence_fragment: str) -> int:
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
        try:
            async with semaphore:
                response = await asyncio.to_thread(
                    gemini_model.generate_content,
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": 0.0}
                )
            return extract_rating(response.text.strip())
        except Exception as e:
            print(f"Warning: fluency scoring failed: {e}")
            return 0
    return _inner

def harmonic_mean(scores):
    """
    Computes the harmonic mean of the provided scores.
    If any score is zero, the harmonic mean will be zero.
    """
    if any(score == 0 for score in scores):
        return 0
    return len(scores) / sum(1.0 / score for score in scores)

async def llm_judge(sentence: str, concept: str, entry_idx: int, sent_idx: int, total_sents: int, evaluate_concept_score, evaluate_fluency_score) -> dict:
    print(f"    [Entry {entry_idx}] Sentence {sent_idx+1}/{total_sents}: evaluating...")
    concept_score = await evaluate_concept_score(concept, sentence)
    fluency_score = await evaluate_fluency_score(sentence)
    final_score = harmonic_mean([concept_score, fluency_score])
    return {
        "sentence_index": sent_idx,
        "steered_sentence": sentence,
        "concept_score": concept_score,
        "fluency_score": fluency_score,
        "final_score": final_score,
    }

async def process_entry(idx: int, entry: dict, concept_map: dict, total_entries: int, is_diffmean=False, evaluate_concept_score=None, evaluate_fluency_score=None) -> dict:
    """
    Processes one steered entry and returns a single dict that includes
    all scores for its sentences under 'sentence_results'.
    """
    level = entry.get("level", 0)
    print(f"Processing entry {idx+1}/{total_entries} (K={entry.get('K', 'SAE')}, layer={entry['layer']}, level={level}, h_row={entry['h_row'] if 'h_row' in entry else entry['index']})")
    key = (entry["K"] if not is_diffmean and ("K" in entry) else "SAE", entry["layer"], level, entry["h_row"] if 'h_row' in entry else entry['index'])
    concept_desc = concept_map.get(key)
    if concept_desc is None:
        print(f"Warning: No concept for {key}")
        sentence_results = []
    else:
        sentences = entry.get("steered_sentences", [])
        total_sents = len(sentences)
        sentence_results = [
            await llm_judge(sentence, concept_desc, idx+1, s_idx, total_sents, evaluate_concept_score, evaluate_fluency_score)
            for s_idx, sentence in enumerate(sentences)
        ]

    return {
        "intervention_sign": entry.get("intervention_sign"),
        "alpha": entry.get("alpha"),
        "kl": entry.get("kl"),
        "K": entry.get("K", "SAE"),
        "layer": entry["layer"],
        "level": level,
        "h_row": entry["h_row"] if 'h_row' in entry else entry['index'],
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
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use (default: gemini-1.5-flash)")
    parser.add_argument("--ranks", required=True, help='K filter, e.g. "100" or "64,100" or "64-128"')
    parser.add_argument("--layers", required=True, help='Layer filter, e.g. "0,8,16" or "0-16"')
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent API calls (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens for each completion (default: 200)")
    parser.add_argument("--retries", type=int, default=5, help="Tenacity stop_after_attempt (default: 5)")
    parser.add_argument(
        "--diffmean",
        action="store_true",
        help="Enable DiffMean baseline"
        )
    parser.add_argument("--api-key-var", default="GEMINI_API_KEY",
                        help="Env var name holding your API key (default: GEMINI_API_KEY)")
    args = parser.parse_args()

    # Load .env and get API key
    load_dotenv()
    api_key = os.getenv(args.api_key_var)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {args.api_key_var}. "
            f"Create a .env with {args.api_key_var}=sk-... or export it in your shell."
        )

    # Initialize global model + semaphore
    global model, semaphore, retries
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    retries = args.retries
    
    # Create evaluator functions
    evaluate_concept_score = make_evaluate_concept_score(args.retries, model, args.max_tokens, semaphore)
    evaluate_fluency_score = make_evaluate_fluency_score(args.retries, model, args.max_tokens, semaphore)

    # Read inputs
    with open(args.input, "r") as f:
        steered_entries = json.load(f)
    with open(args.concepts, "r") as f:
        concepts = json.load(f)

    ranks = parse_int_list(args.ranks)
    layers = parse_int_list(args.layers)
    levels = list(range(len(ranks)))

    # Filter entries
    filtered = [e for e in steered_entries if ("K" not in e or int(e["K"]) in ranks) and int(e["layer"]) in layers and (int(e.get("level", 0)) in levels)]
    total_entries = len(filtered)
    print(f"Selected {total_entries} entries (K in {ranks}, layer in {layers}).")
    # Build concept lookup
    concept_map = {
        (int(c["K"]) if not args.diffmean and  "K" in c else "SAE", int(c["layer"]), int(c.get("level", 0)), int(c['h_row'] if 'h_row' in c else c['index'])): c.get("description", c.get("concept"))
        for c in concepts
        if c.get("description", c.get("concept")) and "TRASH" not in c.get("description", c.get("concept"))
    }

    # Process
    tasks = [
        asyncio.create_task(process_entry(i, entry, concept_map, total_entries, is_diffmean=args.diffmean, evaluate_concept_score=evaluate_concept_score, evaluate_fluency_score=evaluate_fluency_score))
        for i, entry in enumerate(filtered)
    ]
    all_results = await asyncio.gather(*tasks)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Done. Saved results to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
