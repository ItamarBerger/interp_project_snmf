import asyncio
import json
import os
import argparse
from typing import List, Any
from dotenv import load_dotenv
import logging

from experiments.utils import load_existing_jobs, GeminiBatchClient, submit_batches
from utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------- Prompts ---------------- #
CONCEPT_PROMPT = """
You are given a set of tokens, their surrounding context (words before and after the token), and an importance score.
Your task is to identify a brief, direct concept label that describes the connection between these tokens.

### Instructions:

1. **Focus on High-Importance Samples:**
   Examine only the token-context pairs with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring pairs.

2. **Prioritize Tokens First:**
   - First look for patterns in the tokens themselves (identical tokens, related words, grammatical features, suffixes/prefixes, semantic patterns, etc.)
   - Only examine contexts if the tokens themselves show no clear relationship

3. **Output a Concise Label:**
   - Provide ONLY a brief concept label or short phrase
   - Do NOT start with phrases like "The tokens represent..." or "The concept is..."
   - If one specific token dominates, name it directly in quotes
   - Keep it as short as possible while being meaningful

### Input:

Token-Context Pairs:
```{token_context_str}```

Output only the concept label:
"""


# ---------------- Helpers ---------------- #
def extract_results_section(text: str) -> str | None:
    # Since we no longer have a "Results:" section, just return the text directly
    return text.strip() if text else None


def load_data(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_data(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _parse_int_list(csv: str) -> List[int]:
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def _to_float_activation(x) -> float:
    """Handle values like 0.123, string 0.123, or tensor(0.123) safely."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace("tensor(", "").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------- Argparse ---------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate concept descriptions from token-context activations.")
    # I/O
    p.add_argument(
        "--input-json",
        default="rebuttal/init_methods/svd/concept_contexts.json",
        help="Path to input JSON containing top_activations per (K, layer, level, h_row).",
    )
    p.add_argument("--output-json", default="rebuttal/init_methods/svd/input_descriptions.json", help="Path to write the output descriptions JSON.")
    # Gemini
    p.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name.")
    p.add_argument("--env-var", default="GEMINI_API_KEY", help="Environment variable holding the API key (loaded via python-dotenv if present).")
    # Filtering / selection
    p.add_argument("--layers", default="0,6,12,18,25,31", help="Comma-separated list of layer indices to include.")
    p.add_argument("--k-values", default="100", help="Comma-separated list of K (rank) values to include.")
    # Generation controls
    p.add_argument("--top-m", type=int, default=10, help="Number of top activations to consider.")
    p.add_argument("--max-tokens", type=int, default=200, help="max_tokens for each completion.")
    # API controls
    p.add_argument("--batch-size", type=int, default=10000, help="Number of prompts to batch together in a single API job.")
    p.add_argument("--poll-interval", type=int, default=300, help="Polling interval in seconds for batch jobs.")
    p.add_argument("--submitted-jobs-file", type=str, help="Path to file tracking submitted jobs.")
    p.add_argument("--job-backup-folder", default=None, type=str, help="Folder to back up job submissions.")
    return p


def get_prompt_for_entry(entry, top_m: int) -> str:
    # sort and pick top-M by activation
    top = sorted(entry["top_activations"], key=lambda x: _to_float_activation(x["activation"]), reverse=True)[:top_m]

    token_context_str = "\n".join(
        f"Token: `{act['token']}`, Context: `{act['context']}` | Score: `{_to_float_activation(act['activation'])}`" for act in top
    )
    return CONCEPT_PROMPT.format(token_context_str=token_context_str)



def prepare_prompts_and_metadata(filtered_data: list[dict], top_m: int) -> tuple[dict[str, str], dict[str, dict]]:
    prompts_map = {}
    meta_map = {}

    for entry in filtered_data:
        # Create a unique ID for reconstruction
        # Using L{layer}_K{K}_r{h_row} as unique identifier
        uid = f"L{entry['layer']}_K{entry['K']}_LV{entry['level']}_r{entry['h_row']}"

        prompt = get_prompt_for_entry(entry, top_m)
        prompts_map[uid] = prompt
        meta_map[uid] = entry

    logger.info(f"Generated {len(prompts_map)} prompts.")
    return prompts_map, meta_map


def process_results(results_map: dict[str, str], meta_map: dict[str, dict]) -> list[dict]:
    processed_results = []

    for uid, result_text in results_map.items():
        if uid not in meta_map:
            logger.warning(f"Unknown ID in results: {uid}")
            continue

        entry = meta_map[uid]
        description = extract_results_section(result_text)

        processed_results.append(
            {
                "K": entry["K"],
                "layer": entry["layer"],
                "level": entry["level"],
                "h_row": entry["h_row"],
                "description": description
            }
        )

    return processed_results


# ---------------- Main ---------------- #
async def run(args):
    data = load_data(args.input_json)
    layers = set(_parse_int_list(args.layers))
    k_values = set(_parse_int_list(args.k_values))
    # each k corresponds to a specific level. Hence number of levels is len(k_values)
    n_levels = len(k_values)
    levels = [i for i in range(n_levels)]

    # Load API key
    load_dotenv()
    api_key = os.getenv(args.env_var)
    if not api_key:
        raise RuntimeError(f"API key not found in environment variable '{args.env_var}'.")


    # Filter data
    filtered_data = [e for e in data if (int(e["layer"]) in layers and int(e["K"]) in k_values and int(e["level"]) in levels)]

    # Prepare the prompts and metadata for all entries first (before any API calls) to ensure we have a complete mapping
    prompts_map, meta_map = prepare_prompts_and_metadata(filtered_data, args.top_m)

    if not prompts_map:
        logger.warning("No prompts to process after filtering. Exiting.")
        return

    client = GeminiBatchClient(
        api_key=api_key,
        model_name=args.model,
        job_backup_folder=args.job_backup_folder,
        submitted_jobs_path=args.submitted_jobs_file
    )

    generation_config = {
        "max_output_tokens": args.max_tokens,
        "temperature": 0.2,
    }

    existing_jobs = load_existing_jobs(args.submitted_jobs_file)

    active_jobs = await submit_batches(
        prompts_map=prompts_map,
        client=client,
        existing_jobs=existing_jobs,
        submitted_jobs_file=args.submitted_jobs_file,
        batch_size=args.batch_size,
        batch_name_prefix="concept_desc_gen",
        generation_config=generation_config
    )

    logger.info(f"Submitted {len(active_jobs) - len(existing_jobs)} new jobs.")

    # Wait for the results
    logger.info("=== Phase 2: Waiting for batch jobs to complete  ===")
    completed_jobs = await client.wait_for_jobs(active_jobs, poll_interval=args.poll_interval)

    logger.info("=== Phase 3: Retrieve results from client ===")
    results_map = {}
    for job_name in completed_jobs:
        job_results = await client.retrieve_batch_results(job_name)
        results_map.update(job_results)


    logger.info("=== Phase 4: extracting results ===")
    results = process_results(results_map, meta_map)

    save_data(args.output_json, results)
    logger.info(f"\nWrote {len(results)} descriptions to {args.output_json}")


def main():
    setup_logging()
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
