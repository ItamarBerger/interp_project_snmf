import argparse
import json
import os
import logging
import copy
import asyncio
from google import genai
from google.genai._interactions.types import generation_config

from utils import setup_logging
from experiments.utils import GeminiBatchClient, submit_batches, ensure_all_prompts_have_results
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

RESULTS_FILE_NAME = "causal_results_in.json"
TOP_RANKS = ["k400", "k800"]
SUBMITTED_JOBS_FILE = os.path.join("submitted_job_files", "concept_trends_submitted_jobs_{top_ranks}_{models}.json")
PROMPTS_MAP_FILE = os.path.join("experiments", "artifacts", "concept_trends",  "concept_trends_prompts_map_{top_ranks}_{models}.json")
META_MAP_FILE = os.path.join("experiments", "artifacts", "concept_trends",  "concept_trends_meta_map_{top_ranks}_{models}.json")
OUTPUT_FILE = os.path.join("experiments", "artifacts", "concept_trends", "processed_results_{top_ranks}_{models}.json")
JOB_BACKUPS = os.path.join("experiments", "artifacts", "concept_trends", "job_backups", "{top_ranks}_{models}")
MODELS_SYMBOL_MAP = {
    "gpt2-small": "gpt",
    "Llama-3.1-8B": "llama",
    "gemma-2-2b": "gemma",
}
SYNCHRONOUS_API_BACKUP_FILE = os.path.join("experiments", "artifacts", "concept_trends", "synchronous_api_backup_{top_ranks}_{models}.json")
MISMATCHES = os.path.join("experiments", "artifacts", "concept_trends", "mismatches_{top_ranks}_{models}.json")
UNGROUPED_KEY = "<<ungrouped-terms>>"

PROMPT = """
### Instructions
You are given a list of terms. Your task is to identify terms that are identical but written slightly differently and to group these terms together.

Strict Grouping Criteria:
1. Terms must refer to the exact same concept or entity. Do not group non identical concepts together.
2. Do NOT group terms if they differ in:
   - Number (e.g., "I" [singular] vs "We" [plural] must be separate).
   - Case (e.g., "I" [subject] vs "Me" [object] must be separate).
   - Entity (e.g. "Tree" and "Branch" should be separate)
   - Part of Speech (e.g., "Comparative Adjective" vs "Comparative Adverb" must be separate).
3. Do not create "catch-all" categories. If a term is a specific word (e.g., "it") and another is a functional description (Anaphoric "it"), they only group if they refer to the same specific usage.
4. The results must be mutually exclusive. Every original term string must be preserved character-for-character.
5. If there are terms that do not share anything in common with other terms add them to a list in the dictionary under a special key marked {ungrouped_key}.
6. All original terms MUST be present in exactly one dict in the output.

### Output Format
Analysis:
<A brief explanation of how you distinguished between similar but grammatically different terms.>

Results:
<A JSON-parsable dictionary where keys are short, descriptive group names and values are lists of strings. Even if there is a single value, it should be represented as a list (of a single item). Ensure 100% character accuracy from the source list.>

### List of terms
```{terms}```
"""

def generate_prompt(terms: list[str]) -> str:
    return PROMPT.format(terms=terms, ungrouped_key=UNGROUPED_KEY)

def traverse_and_load_files(base_path,  top_ranks: list[str], models: list[str] | None = None) -> dict:
    results = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "causal_results_in.json":
                file_path = Path(str(os.path.join(root, file)))
                model_name = file_path.parent.name

                # Skip this file if not specified in the models filter
                if models and not model_name in models:
                    continue

                top_rank = file_path.parent.parent.name
                if top_rank not in TOP_RANKS:
                    # Hack for llama
                    top_rank = file_path.parent.parent.parent.name

                if top_rank not in top_ranks:
                    continue

                if top_rank and not top_rank in results:
                    results[top_rank] = {}
                logger.info(f"Loading results from {file_path} (model: {model_name})")
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Organize data by layer and level
                organized_data = {}
                for entry in data:
                    layer = entry["layer"]
                    level = entry["level"]

                    if layer not in organized_data:
                        organized_data[layer] = defaultdict(list)

                    organized_data[layer][level].append(entry)

                results[top_rank][model_name] = organized_data

    return results


def get_prompt_id(model_name: str, top_rank: str, layer: int, level: int) -> str:
    return f"{model_name}_top{top_rank}_L{layer}_LV{level}"

def generate_prompts_map(data: dict) -> tuple[dict, dict]:
    prompts_map = {}
    meta_map = {}
    for top_rank, models_data in data.items():
        for model_name, layer_data in models_data.items():
            for layer, levels_data in layer_data.items():
                for level in sorted(levels_data.keys()):
                    descriptions_per_level = set(entry["description"] for entry in levels_data[level])
                    prompt = generate_prompt(list(descriptions_per_level))
                    prompt_id = get_prompt_id(model_name, top_rank, layer, level)
                    prompts_map[prompt_id] = prompt
                    meta_map[prompt_id] = {
                        "model_name": model_name,
                        "top_rank": top_rank,
                        "layer": layer,
                        "level": level,
                        "descriptions": descriptions_per_level
                    }
    return prompts_map, meta_map


def parse_str_list(s: str) -> list[str]:
    return [item.strip() for item in s.split(",")]

def get_model_str(args) -> str:
    if not args.models:
        return "all_models"
    return "_".join([MODELS_SYMBOL_MAP.get(model, model) for model in args.models])

def save_prompts_and_meta(prompts_map: dict, meta_map: dict, args):
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)

    prompts_map_file = PROMPTS_MAP_FILE.format(top_ranks=top_ranks_str, models=models_str)
    meta_map_file = META_MAP_FILE.format(top_ranks=top_ranks_str, models=models_str)

    with open(prompts_map_file, "w") as f:
        json.dump(prompts_map, f, indent=2)
    with open(meta_map_file, "w") as f:
        json.dump({k: list(v) for k, v in meta_map.items()}, f, indent=2)

def get_submitted_jobs_file(args) -> str:
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)
    return SUBMITTED_JOBS_FILE.format(top_ranks=top_ranks_str, models=models_str)

def get_backup_folder(args) -> str:
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)
    return JOB_BACKUPS.format(top_ranks=top_ranks_str, models=models_str)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze concept trends across layers.")
    parser.add_argument("--base-path", type=str, required=True, help="Path to the root directory where the artifacts and the causal_results_in.json files are stored.")
    parser.add_argument("--models", type=parse_str_list, default=None,
                        help="Comma-separated list of model names to include (e.g., 'gpt2-small,Llama-3.1-8B'). If omitted, all models will be included.")
    parser.add_argument("--top-ranks", type=parse_str_list, default=TOP_RANKS, help="Comma-separated list of top ranks to include (e.g., 'k400,k800'). Default is 'k400,k800'.")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite", help="Gemini model name to use for analysis.")
    parser.add_argument("--batch-size", type=int, default=10000, help="Number of prompts to batch together in a single API job.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds for batch jobs.")
    parser.add_argument("--use-batch-api", action="store_true", help="Whether to use the batch API for submitting jobs to Gemini.")
    return parser.parse_args()


def extract_response_json(response: str) -> dict:
    try:
        # Attempt to find the JSON part of the response
        json_start = response.find("```json") + len("```json")
        json_end = response.rfind("```")
        if json_start == -1 or json_end == -1:
            raise ValueError("No JSON object found in the response.")

        json_str = response[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error extracting JSON from response: {e}")
        return {}

def validate_result(result: dict, descriptions: set) -> bool:
    returned_group_values = set(v for group in result.values() for v in group)
    if not returned_group_values == descriptions:
        return False
    return True


def fix_unwrapped_values(result: dict) -> None:
    """
    In some cases, the model may return values that are not wrapped in a list, even if they are the only value in their group. This function checks for such cases and wraps them in a list to ensure consistency in the output format.
    """
    for group, values in result.items():
        if isinstance(values, str):
            logger.warning(f"Group '{group}' has a single value that is not wrapped in a list: '{values}'. Wrapping this value in a list for consistency.")
            result[group] = [values]


def clean_duplicates(result: dict) -> dict:
    fixed_results = copy.deepcopy(result)

    # Clean duplicates from each group first
    for group, values in fixed_results.items():
        unique_values = set(values)
        if len(unique_values) < len(values):
            logger.warning(f"Group '{group}' has duplicate values: {set([v for v in values if values.count(v) > 1])}. Removing duplicates from this group.")
            fixed_results[group] = list(unique_values)

    # Clean duplicate values from the ungrouped category if they are already present in other groups
    ungrouped_values = set(result.get(UNGROUPED_KEY, []))
    to_remove_from_ungrouped = set()
    for group, values in fixed_results.items():
        if group == UNGROUPED_KEY:
            continue
        values_set = set(values)
        duplicates = values_set.intersection(ungrouped_values)
        to_remove_from_ungrouped.update(duplicates)

    if to_remove_from_ungrouped:
        num_to_remove = len(to_remove_from_ungrouped)
        fixed_results[UNGROUPED_KEY] = [v for v in fixed_results.get(UNGROUPED_KEY, []) if v not in to_remove_from_ungrouped]
        logger.info("Removed %d duplicate values from %s group that were already present in other groups: %s", num_to_remove, UNGROUPED_KEY, to_remove_from_ungrouped)

    # If there are still any duplicates in other groups, just remove them from one of those groups
    entries_by_groups = defaultdict(list)
    for group, values in fixed_results.items():
        for value in values:
            if group in entries_by_groups.get(value, []):
               raise "This really should not happen, this means that we have the same value duplicated in the same group even after cleaning, which should be impossible. Please investigate."
            entries_by_groups[value].append(group)

    duplicate_entries = {value: groups for value, groups in entries_by_groups.items() if len(groups) > 1}
    for value, groups in duplicate_entries.items():
        logger.warning(f"Value '{value}' is duplicated across groups {groups}. Removing randomly from all but the first group.")
        # remove from all groups except the first one
        for group in groups[1:]:
            if value in fixed_results[group]:
                fixed_results[group].remove(value)
                if not fixed_results[group]:
                    logger.warning(f"Group '{group}' is now empty after removing duplicate value '{value}'. Deleting this group from results.")
                    del fixed_results[group]
    return fixed_results



def fix_results(result: dict, descriptions: set) -> dict:
    """
    The model sometimes misses items in the results. This is not ideal, but in such case, we add them to the
    UNGROUPED_KEY category to ensure all descriptions are accounted for in the final results.
    """
    # Find discrepancies
    returned_group_values = set(v for group in result.values() for v in group)

    # Make a copy of the original result to modify while iterating
    fixed_results = clean_duplicates(result)

    # first remove extra values that the model added but are not in the descriptions (if any)
    extra_values = returned_group_values - descriptions
    for group, values in result.items():
        if group not in fixed_results:
            # This means that this group was already removed due to duplicates, so we can skip it
            continue
        values_without_extra = set(values) - extra_values
        if len(values_without_extra) < len(values):
            logger.warning(f"Group '{group}' has extra values that are not in descriptions: {extra_values}. Removing these from the group.")
        # if values_without extra is empty, remove the key entirely
        if not values_without_extra:
            logger.warning(f"Deleting group {group} from results.")
            del fixed_results[group]
        else:
            fixed_results[group] = list(values_without_extra)

    # Add missing values
    missing_values = descriptions - set(v for group in fixed_results.values() for v in group)
    if missing_values:
        logger.warning(f"Result is missing values: {missing_values}. Adding them to {UNGROUPED_KEY} category.")
        if UNGROUPED_KEY not in fixed_results:
            fixed_results[UNGROUPED_KEY] = []
            logger.info(f"Had to create {UNGROUPED_KEY} category to account for missing values.")
        fixed_results[UNGROUPED_KEY].extend(list(missing_values))

    return fixed_results


def re_organize_result(result: dict)-> dict:
    """
    Removes the UNGROUPED_KEY category and re-assigns its values to their own group with the same name as the value.
     This is done to make the results more consistent and easier to analyze for trends, since we want to treat ungrouped terms as their own individual groups in the analysis.
    """
    organized_result = copy.deepcopy(result)
    if UNGROUPED_KEY in organized_result:
        ungrouped_values = organized_result.pop(UNGROUPED_KEY)
        for value in ungrouped_values:
            if value in organized_result:
                logger.warning(f"Value {value} already exists. We assume we can extend the group, but you should verify")
                organized_result[value].append(value)
            else:
                organized_result[value] = [value]

    return organized_result

def process_and_validate_results(results_map: dict, meta_map: dict):
    processed_results = {}
    for prompt_id, result in results_map.items():
        descriptions = meta_map.get(prompt_id, {}).get("descriptions", set())
        results = extract_response_json(result)

        if not results:
            logger.warning(f"No valid JSON results for prompt_id {prompt_id}. Skipping.")
            continue


        if not validate_result(results, descriptions):
            logger.warning(f"Mismatch in descriptions and returned values for prompt_id {prompt_id}. Descriptions: {descriptions}, Returned: {returned_group_values}")
            continue

        processed_results[prompt_id] = {
            "model_name": meta_map[prompt_id]["model_name"],
            "top_rank": meta_map[prompt_id]["top_rank"],
            "layer": meta_map[prompt_id]["layer"],
            "level": meta_map[prompt_id]["level"],
            "descriptions": descriptions,
            "grouping": results
        }
    return processed_results

def save_processes_results(processed_results: dict, args):
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)
    output_file = OUTPUT_FILE.format(top_ranks=top_ranks_str, models=models_str)

    # Organize the results in a dict by model_name -> { top_rank -> {layer -> list of level data}} for easier analysis later on
    final_results = {}

    for prompt_id, data in processed_results.items():
        model_name = data["model_name"]
        top_rank = data["top_rank"]
        layer = data["layer"]

        if model_name not in final_results:
            final_results[model_name] = {}
        if top_rank not in final_results[model_name]:
            final_results[model_name][top_rank] = defaultdict(list)

        final_results[model_name][top_rank][layer].append(data)


    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved processed results to {output_file}")


async def submit_and_retrieve_results(prompts_map: dict, meta_map: dict, args, gen_config: dict) -> dict:
    gemini_batch_client = GeminiBatchClient(
        api_key=os.environ["GEMINI_API_KEY"],
        model_name=args.gemini_model,
        submitted_jobs_path=get_submitted_jobs_file(args),
        job_backup_folder=get_backup_folder(args)
    )

    # Submit the jobs to Gemini
    existing_jobs = gemini_batch_client.load_submitted_jobs()

    active_jobs = await submit_batches(
        prompts_map=prompts_map,
        client=gemini_batch_client,
        existing_jobs=existing_jobs,
        submitted_jobs_file=get_submitted_jobs_file(args),
        batch_size=args.batch_size,
        batch_name_prefix="concept_trends_analysis",
        generation_config=gen_config
    )

    # Wait for results
    logger.info("Waiting for Gemini jobs to complete...")
    completed_jobs = await gemini_batch_client.wait_for_jobs(active_jobs, poll_interval=args.poll_interval)

    # Retrieve results
    results_map = {}
    for job_name in completed_jobs:
        job_results = await gemini_batch_client.retrieve_batch_results(job_name)
        results_map.update(job_results)

    # Process results
    processed_results = process_and_validate_results(results_map, meta_map)

    ensure_all_prompts_have_results(prompts_map, processed_results)

    return processed_results


def update_file_with_sync_api_results(result: dict, args):
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)
    backup_file = SYNCHRONOUS_API_BACKUP_FILE.format(top_ranks=top_ranks_str, models=models_str)

    if os.path.exists(backup_file):
        with open(backup_file, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    existing_results.update(result)

    # Create parent folders for backup file if they don't exist
    os.makedirs(os.path.dirname(backup_file), exist_ok=True)

    with open(backup_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    logger.info(f"Updated synchronous API results backup file at {backup_file}")


def load_file(args, file_path_template: str) -> dict:
    top_ranks_str = "_".join(args.top_ranks)
    models_str = get_model_str(args)
    file_path = file_path_template.format(top_ranks=top_ranks_str, models=models_str)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    else:
        logger.info(f"No existing file found at {file_path}. Starting fresh.")
        return {}

def load_sync_api_results(args) -> dict:
    return load_file(args, SYNCHRONOUS_API_BACKUP_FILE)

def load_mismatches(args) -> dict:
    return load_file(args, MISMATCHES)


def make_synchronous_calls(prompts_map: dict, meta_map: dict, args, gen_config: dict) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    processed_results = {}
    total_calls = len(prompts_map)
    # load existing results from backup file to avoid redundant API calls
    results_map = load_sync_api_results(args)
    mismatches = load_mismatches(args)
    for index, (prompt_id, prompt) in enumerate(prompts_map.items()):
        try:
            if prompt_id in results_map and not prompt_id in mismatches:
                logger.info(f"Found existing result for prompt_id {prompt_id} from backup file. Using this.")
                result = results_map[prompt_id]
            else:
                if prompt_id in mismatches:
                    logger.info(f"Prompt_id {prompt_id} was previously marked as mismatch. Re-evaluating this prompt.")
                logger.info("Making call for prompt_id: %s [%s/%s]", prompt_id, index + 1, total_calls)

                response = client.models.generate_content(
                    model=args.gemini_model,
                    contents=[prompt],
                    config=genai.types.GenerateContentConfig(
                        **gen_config
                    )
                )

                result = response.text

                # update the backup file with the new result after each call to ensure progress is saved
                update_file_with_sync_api_results({prompt_id: result}, args)

            grouping = extract_response_json(result)

            if not grouping:
                logger.warning(f"No valid JSON results for prompt_id {prompt_id}. Skipping.")
                continue

            descriptions = meta_map[prompt_id]["descriptions"]

            fix_unwrapped_values(grouping)
            grouping = fix_results(grouping, descriptions)
            grouping = re_organize_result(grouping)

            # Sanity check to see if the results are valid after fixing
            if not validate_result(grouping, meta_map[prompt_id]["descriptions"]):
                logger.warning(f"After fixing, still mismatch in descriptions and returned values for prompt_id {prompt_id}. Descriptions: {meta_map[prompt_id]['descriptions']}, Returned: {set(v for group in grouping.values() for v in group)}. Marking this prompt as mismatch for further analysis.")
                mismatches[prompt_id] = {
                    "descriptions": list(descriptions),
                    "grouping":  grouping
                }
                continue

            processed_results[prompt_id] = {
                "model_name": meta_map[prompt_id]["model_name"],
                "top_rank": meta_map[prompt_id]["top_rank"],
                "layer": meta_map[prompt_id]["layer"],
                "level": meta_map[prompt_id]["level"],
                "descriptions": list(descriptions),
                "grouping": grouping
            }
        except Exception as e:
            logger.error(f"Error processing prompt_id {prompt_id}: {e}")
            raise e
    if mismatches:
        # save to file
        top_ranks_str = "_".join(args.top_ranks)
        models_str = get_model_str(args)
        mismatches_file = MISMATCHES.format(top_ranks=top_ranks_str, models=models_str)
        with open(mismatches_file, "w") as f:
            json.dump(mismatches, f, indent=2)
        logger.info(f"Saved mismatches to {mismatches_file}")

    return processed_results


async def main():
    setup_logging()
    args = parse_args()

    # Traverse and load the files
    data = traverse_and_load_files(
        base_path=args.base_path,
        top_ranks=args.top_ranks,
        models=args.models
    )

    # Prepare the prompts and meta map
    prompts_map, meta_map = generate_prompts_map(data)

    # Config
    gen_config = {
        "temperature": 0.0,
    }

    # Save the prompts and metadata for later use
    save_prompts_and_meta(prompts_map, meta_map, args)

    if args.use_batch_api:
        # Submit jobs to Gemini and retrieve results
        processed_results = await submit_and_retrieve_results(prompts_map, meta_map, args, gen_config)
    else:
        processed_results = make_synchronous_calls(prompts_map, meta_map, args, gen_config)


    # Save processed results
    save_processes_results(processed_results, args)



if __name__ == "__main__":
    asyncio.run(main())