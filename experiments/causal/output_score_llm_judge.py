import asyncio
import json
import re
import os
import sys
import argparse
import time
from typing import List, Generator, Tuple
from dotenv import load_dotenv
import logging

from experiments.causal.judge_utils import extract_rating, load_existing_jobs, submit_batches
from experiments.utils import GeminiBatchClient
from utils import setup_logging

logger = logging.getLogger(__name__)

# ------------------------------
# Helpers
# ------------------------------
def parse_int_list(spec: str) -> List[int]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def get_concept_prompt(concept: str, sentence_fragment: str) -> str:
    return f"""Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
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


def get_fluency_prompt(sentence_fragment: str) -> str:
    return  f"""Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability.
After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect.
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence_fragment}
[Sentence Fragment End]"""

def harmonic_mean(scores: List[int]) -> float:
    if any(s == 0 for s in scores):
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)


def process_entries(filtered: list, concept_map_dict: dict, sparsity: str):
    processed_structure = []
    meta_map = {}
    prompts_map = {}

    number_of_concepts_not_found = 0

    for idx, entry in enumerate(filtered):
        # We never want to default to 0, this must have been something
        # related to SAEs etc
        level = entry["hier_level"]
        h_row = entry['h_row']
        layer = entry["layer"]
        k = int(entry["K"])
        intervention_sign = entry.get("intervention_sign")

        # Important! output_score_llm_judge uses sign in the key
        # Because the vocab projections contained negative and positive tokens
        key = (k, layer, level, h_row, intervention_sign)
        concept_desc = concept_map_dict.get(key)

        kl = entry.get("kl")
        alpha = entry.get("alpha")

        # Prepare result skeleton
        entry_result = {
            "intervention_sign": intervention_sign,
            "alpha": alpha,
            "kl": kl,
            "K": k,
            "layer": layer,
            "level": level,
            "h_row": h_row,
            "description": concept_desc,
            "sparsity": sparsity,
            "sentence_results": []
        }

        if concept_desc:
            sentences = entry.get("steered_sentences", [])
            for s_idx, sentence in enumerate(sentences):
                # Unique IDs
                c_id = f"L{layer}_LV{level}_K{k}_r{h_row}_kl{kl}_IS{intervention_sign}_s{s_idx}_c"
                f_id = f"L{layer}_LV{level}_K{k}_r{h_row}_kl{kl}_IS{intervention_sign}_s{s_idx}_f"

                # Add to prompts map
                prompts_map[c_id] = get_concept_prompt(concept_desc, sentence)
                prompts_map[f_id] = get_fluency_prompt(sentence)

                # Prepare placeholder object
                sent_obj = {
                    "sentence_index": s_idx,
                    "steered_sentence": sentence,
                    "concept_score": None,
                    "fluency_score": None,
                    "final_score": None
                }
                entry_result["sentence_results"].append(sent_obj)

                # Map IDs to the result object
                meta_map[c_id] = (sent_obj, "concept_score")
                meta_map[f_id] = (sent_obj, "fluency_score")
        else:
            number_of_concepts_not_found += 1
            logger.warning(f"⚠ Warning: No concept for {key}")

        processed_structure.append(entry_result)

    logger.info(f"Processed {len(filtered)} entries. Concepts not found for {number_of_concepts_not_found} entries.")
    return processed_structure, meta_map, prompts_map


# ------------------------------
# Main
# ------------------------------
async def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Score steered sentences (output-centric) for concept coverage and fluency.")
    parser.add_argument("--input", required=True, help="Path to steered entries JSON")
    parser.add_argument("--concepts", required=True, help="Path to concepts JSON")
    parser.add_argument("--output", required=True, help="Where to write aggregated results JSON")
    parser.add_argument("--ranks", required=True, help='K filter, e.g. "100" or "64,100" or "64-128"')
    parser.add_argument("--layers", required=True, help='Layer filter, e.g. "23,31" or "0-16"')
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model (default: gemini-2.0-flash)")
    # [CHANGED] Arguments for batch processing
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for API (default: 10000)")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Polling interval in seconds (default: 300 seconds = 5 minutes)")
    parser.add_argument("--sparsity", default="s0.05", help="Sparsity tag to include in results (default: s0.05)")
    parser.add_argument("--api-key-var", default="GEMINI_API_KEY",
                        help="Env var containing the API key (default: GEMINI_API_KEY)")
    parser.add_argument("--submitted-jobs-file", type=str,
                        help="In case the execution failed - this file holds the list of already submitted jobs to avoid resubmission.")
    parser.add_argument("--job-backup-folder", default=None, type=str, help="Folder to back up job submissions.")
    args = parser.parse_args()

    logger.info("=== LLM JUDGE (Batch Processing - Output Centric) ===")

    # Load .env and get API key
    load_dotenv()
    api_key = os.getenv(args.api_key_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in {args.api_key_var}")

    # Read inputs
    logger.info("Loading input files...")
    with open(args.input, "r") as f:
        steered_entries = json.load(f)
    with open(args.concepts, "r") as f:
        concepts = json.load(f)

    ranks = parse_int_list(args.ranks)
    layers = parse_int_list(args.layers)

    # Filter entries
    logger.info("Filtering entries...")
    filtered = [e for e in steered_entries if ("K" not in e or int(e["K"]) in ranks) and int(e["layer"]) in layers]

    # Sort filtered entries by layer, level, K, h_row, kl, intervention_sign
    filtered.sort(key=lambda e: (int(e["layer"]), int(e["hier_level"]), int(e["K"]), int(e['h_row']), float(e["kl"]), e["intervention_sign"]))

    total_entries = len(filtered)
    logger.info(f"Selected {total_entries} entries out of {len(steered_entries)}")

    # Build lookup: (K, layer, level, h_row, sign) -> description  (skip TRASH)
    concept_map = {
        (int(c["K"]), int(c["layer"]), int(c["level"]),
         int(c["h_row"]), c["sign"]): c["description"]
        for c in concepts
        if c.get("description") and "TRASH" not in c["description"]
    }

    # Generate prompts
    logger.info("Generating prompts and building maps...")
    # [CHANGED] Call process_entries instead of running async loop
    processed_structure, meta_map, prompts_map = process_entries(filtered, concept_map, args.sparsity)

    if not prompts_map:
        logger.warning("No prompts generated (check filters or concept map). Exiting.")
        return

    existing_jobs = load_existing_jobs(args)
    # Submit jobs
    client = GeminiBatchClient(api_key=api_key, model_name=args.model, submitted_jobs_path=args.submitted_jobs_file,
                              job_backup_folder=args.job_backup_folder)
    active_jobs = await submit_batches(prompts_map, client, existing_jobs, args, judge_type="output")

    # Wait for jobs to complete
    logger.info("Phase 2: Waiting for jobs to complete...")
    completed_jobs = await client.wait_for_jobs(active_jobs, poll_interval=args.poll_interval)

    # Retrieve
    logger.info("Phase 3: Retrieving results...")
    results_map = {}
    for job_name in completed_jobs:
        batch_results = await client.retrieve_batch_results(job_name)
        results_map.update(batch_results)

    # Process Scores
    logger.info("Calculating final scores...")
    missed_count = 0
    for rid, result_text in results_map.items():
        if rid in meta_map:
            sent_obj, field = meta_map[rid]
            sent_obj[field] = extract_rating(result_text)
        else:
            logger.debug(f"Unknown ID in results: {rid}")

    # Final Aggregation
    for entry in processed_structure:
        for sent in entry["sentence_results"]:
            if sent["concept_score"] is None: missed_count += 1
            if sent["fluency_score"] is None: missed_count += 1

            sent["final_score"] = harmonic_mean([sent["concept_score"], sent["fluency_score"]])

    if missed_count > 0:
        logger.warning(f"⚠ {missed_count} scores were missing (defaulted to 0).")

    # Save results
    logger.info(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(processed_structure, f, indent=2)
    logger.info("✓ DONE")

if __name__ == "__main__":
    asyncio.run(main())
