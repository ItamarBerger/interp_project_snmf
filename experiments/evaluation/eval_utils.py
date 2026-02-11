from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import sys

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> Dict[str, Any]|List[Dict[str, Any]]:
    """
    Loads JSON data from the specified file path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from file: {file_path}")
        sys.exit(1)


def create_path_if_not_exists(object_path: str, is_file: bool):
    """
    Creates the directory path for the given file path if it does not exist.
    """
    f = Path(object_path)
    if is_file:
        f.parent.mkdir(parents=True, exist_ok=True)
    else:
        f.mkdir(parents=True, exist_ok=True)

def calculate_entry_means(entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the mean concept, fluency, and final scores for a single entry.
    Returns a dictionary with the calculated means.
    """
    results = entry.get('sentence_results', [])
    if not results:
        logger.warning(f"No sentence_results found for entry: {entry.get('description', 'Unknown')}")
        return {
            "mean_concept_score": 0.0,
            "mean_fluency_score": 0.0,
            "mean_final_score": 0.0
        }

    total_concept = sum((r.get('concept_score') or 0) for r in results)
    total_fluency = sum((r.get('fluency_score') or 0) for r in results)
    total_final = sum((r.get('final_score') or 0) for r in results)
    count = len(results)

    return {
        "mean_concept_score": total_concept / count,
        "mean_fluency_score": total_fluency / count,
        "mean_final_score": total_final / count,
        "intervention_sign": entry.get('intervention_sign'),
        "description": entry.get('description'),
        "K": entry.get('K'),
    }


def get_best_results_and_organize_by_layer(grouped_entries: Dict[str, Any], use_extended_keys: bool = False, use_dict: bool = False) -> Dict[str, Any]:
    """
    For each (layer, level, h_row) group, find the maximum mean scores
    across different runs (kl/alpha/sign), then organize results by layer.
    Assumes grouped_entries is a dictionary with keys as (layer, level, h_row) tuples
    Args:
        grouped_entries: a dictionary with keys as (layer, level, h_row) tuples
        use_extended_keys: whether to organize results by layer
        use_dict: Instead of returning a list of dicts per layer, it will return { layer: { (level, h_row): <result> } }

    """

    final_results_by_layer_dict = defaultdict(dict)
    final_results_by_layer = defaultdict(list)

    for (layer, level, h_row), values in grouped_entries.items():
        if not values:
            continue
        # Find max values across all runs (different kl/alpha/sign) for this specific group
        max_mean_concept = max(v['mean_concept_score'] for v in values)
        max_mean_fluency = max(v['mean_fluency_score'] for v in values)
        max_mean_final = max(v['mean_final_score'] for v in values)

        result_entry = {
            "level": level,
            "h_row": h_row,
            "max_mean_concept_score": max_mean_concept,
            "max_mean_fluency_score": max_mean_fluency,
            "max_mean_final_score": max_mean_final,
        }

        if use_extended_keys:
            descriptions, Ks = set([v['description'] for v in values]), set([v['K'] for v in values])
            if len(descriptions) > 1:
                raise ValueError(f"Multiple descriptions found for (layer={layer}, level={level}, h_row={h_row}): {descriptions}. Can't build extended data")
            if len(Ks) > 1:
                raise ValueError(f"Multiple K values found for (layer={layer}, level={level}, h_row={h_row}): {Ks}. Can't build extended data")
            result_entry["description"] = descriptions.pop()
            result_entry["K"] = Ks.pop()


        if use_dict:
            final_results_by_layer_dict[layer][(level, h_row)] = result_entry
        else:
            final_results_by_layer[layer].append(result_entry)

    return final_results_by_layer if not use_dict else final_results_by_layer_dict


def filter_and_aggregate_entries(data: List[Dict[str, Any]], target_layers: Optional[List[int]]) -> tuple[Dict[str, Any], Dict[str, set]]:

    # Dictionary to hold lists of processed entries keyed by (layer, level, h_row)
    grouped_data = defaultdict(list)

    skipped_descriptions_keys = set()
    # Step 1: Filter and Aggregate
    for entry in data:
        # Filter by layer
        layer = entry.get('layer')
        if target_layers and layer not in target_layers:
            continue

        # Skip null descriptions - this could happen for output descriptions
        # Because we split negative and output descriptions
        # We want to take the variation that succeeded (according to intervention_sign)
        if entry.get('description') is None:
            skipped_descriptions_keys.add((layer, entry.get('level'), entry.get('h_row')))
            continue

        # Calculate means
        means = calculate_entry_means(entry)

        # Group key
        level = entry.get('level')
        h_row = entry.get('h_row')
        group_key = (layer, level, h_row)

        grouped_data[group_key].append(means)

    # Sanity check: make sure that for each skipped description, we only took one intervention_sign
    # Since this could only happen for output descriptions for which we generated different description according the intervention_sign
    completely_missing_keys = 0
    layers_with_missing_keys = defaultdict(set)
    for key in skipped_descriptions_keys:
        key_data = grouped_data.get(key, [])
        if not key_data:
            logger.debug(f"No data found for skipped description key!: {key}")
            completely_missing_keys += 1
            layers_with_missing_keys[key[0]].add(key)
        else:
            intervention_signs = set([e["intervention_sign"] for e in key_data])
            if len(intervention_signs) > 1:
                logger.warning(f"Multiple intervention signs found for skipped description key {key}: {intervention_signs}")

    total_unique_keys = len(set(grouped_data.keys())) + completely_missing_keys
    if completely_missing_keys > 0:
        logger.warning(f"Number of missing keys (skipped (layer, level, feature) combinations): {completely_missing_keys} out of {total_unique_keys}")
        for layer, values in layers_with_missing_keys.items():
            count = len(values)
            logger.warning(f"Layer {layer} has {count} missing keys")
        logger.info("You might want to use input descriptions for layers with high missing keys count")


    return grouped_data, layers_with_missing_keys
