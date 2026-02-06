import asyncio
import json
import os
import argparse
from typing import List
from dotenv import load_dotenv
import logging
from experiments.causal.judge_utils import extract_rating, submit_batches, load_existing_jobs
from experiments.utils import GeminiBatchClient
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


# ------------------------------
# Prompts
# ------------------------------

def get_concept_prompt(concept: str, sentence_fragment: str) -> str:
    return  f"""Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
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
    return f"""Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability.
After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect.
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence_fragment}
[Sentence Fragment End]"""


def process_entries(filtered: list, concept_map_dict: dict):
    processed_structure = []
    meta_map = {}
    prompts_map = {}
    for idx, entry in enumerate(filtered):
        level = entry["hier_level"] # Must be present
        h_row = entry['h_row']
        layer = entry["layer"]
        k = entry["K"]
        key = (k, layer, level, h_row)
        concept_desc = concept_map_dict.get(key)

        kl = entry["kl"]
        intervention_sign = entry["intervention_sign"]


        # Prepare result skeleton
        entry_result = {
            "intervention_sign": intervention_sign,
            "alpha": entry.get("alpha"),
            "kl": kl,
            "K": k,
            "layer": layer,
            "level": level,
            "h_row": h_row,
            "description": concept_desc,
            "sentence_results": []
        }

        if concept_desc:
            sentences = entry.get("steered_sentences", [])
            for s_idx, sentence in enumerate(sentences):
                # Create unique IDs so we can use this map to find the correct results
                # This should not depend on the idx - we want to be able to reconstruct the maps if we change the filtering or order of entries
                # !! Important !! do not change this if you have backup files with these ids
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

                # Map IDs to the result object for O(1) update later
                meta_map[c_id] = (sent_obj, "concept_score")
                meta_map[f_id] = (sent_obj, "fluency_score")
        else:
            logger.warning(f"⚠ Warning: No concept for {key}")
        processed_structure.append(entry_result)

    logger.info("built processed structure with %d entries and %d prompts", len(processed_structure), len(prompts_map))
    return processed_structure, meta_map, prompts_map



def load_data(args):
    logger.info("Loading input files...")
    with open(args.input, "r") as f:
        steered_entries = json.load(f)
    with open(args.concepts, "r") as f:
        concepts = json.load(f)

    return steered_entries, concepts


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
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for API (default: 10000)")
    parser.add_argument("--poll-interval", type=int, default=300, help="Polling interval in seconds (default: 300 seconds = 5 minutes)")
    parser.add_argument("--api-key-var", default="GEMINI_API_KEY",
                        help="Env var name holding your API key (default: GEMINI_API_KEY)")
    parser.add_argument("--submitted-jobs-file", type=str, help="In case the execution failed - this file holds the list of already submitted jobs to avoid resubmission.")
    parser.add_argument("--job-backup-folder", default=None, type=str, help="Folder to back up job submissions.")
    args = parser.parse_args()

    logger.info("=== LLM JUDGE (Batch Processing) ===")
    logger.info("Using backup folder: %s", args.job_backup_folder)

    load_dotenv() # for local development
    api_key = os.getenv(args.api_key_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in {args.api_key_var}")


    steered_entries, concepts = load_data(args)

    #  filter and map concepts
    ranks = parse_int_list(args.ranks)
    layers = parse_int_list(args.layers)
    levels = [i for i in range(len(ranks))] if ranks else [0]

    filtered = [
        e for e in steered_entries
        if ("K" not in e or int(e["K"]) in ranks)
           and int(e["layer"]) in layers
           and int(e["hier_level"]) in levels
    ]
    logger.info(f"Filtering: selected {len(filtered)} entries out of {len(steered_entries)}.")

    # Sort filtered entries by layer, level, K, h_row, kl, intervention_sign
    filtered.sort(key=lambda e: (int(e["layer"]), int(e["hier_level"]), int(e["K"]), int(e['h_row']), float(e["kl"]), e["intervention_sign"]))

    concept_map_dict = {
        (int(c["K"]), int(c["layer"]), int(c["level"]),
         int(c['h_row'] if 'h_row' in c else c.get('index', 0))): c.get("description", c.get("concept"))
        for c in concepts
        if c.get("description", c.get("concept")) and "TRASH" not in c.get("description", c.get("concept"))
    }

    #  Generate prompts
    logger.info("Generating prompts and building maps...")
    processed_structure, meta_map, prompts_map = process_entries(filtered, concept_map_dict)


    if not prompts_map:
        logger.warning("No prompts generated (check filters or concept map). Exiting.")
        return

    existing_jobs = load_existing_jobs(args)
    # Submit jobs
    client = GeminiBatchClient(api_key=api_key, model_name=args.model, job_backup_folder=args.job_backup_folder, submitted_jobs_path=args.submitted_jobs_file)
    active_jobs = await submit_batches(prompts_map, client, existing_jobs, args, judge_type="input")

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

    # 6. Final Aggregation
    for entry in processed_structure:
        for sent in entry["sentence_results"]:
            if sent["concept_score"] is None: missed_count += 1
            if sent["fluency_score"] is None: missed_count += 1

            sent["final_score"] = harmonic_mean([sent["concept_score"], sent["fluency_score"]])

    if missed_count > 0:
        logger.warning(f"⚠ {missed_count} scores were missing (defaulted to 0).")

    # 7. Save
    logger.info(f"Saving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(processed_structure, f, indent=2)
    logger.info("✓ DONE")

if __name__ == "__main__":
    asyncio.run(main())
