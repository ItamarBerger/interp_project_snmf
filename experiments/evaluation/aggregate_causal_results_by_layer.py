import argparse
import json
import logging
import sys
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional

from experiments.evaluation.eval_utils import calculate_entry_means, load_data, create_path_if_not_exists
from utils import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process experiment results from a JSON file.")
    parser.add_argument(
        '--input-file',
        type=str,
        help="Path to the input JSON file containing experiment results."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help="Path to save the processed JSON output."
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help="List of layer integers to filter and process. If not provided, all layers are processed."
    )
    return parser.parse_args()



def get_best_results_and_organize_by_layer(grouped_entries: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each (layer, level, h_row) group, find the maximum mean scores
    across different runs (kl/alpha/sign), then organize results by layer.
    Assumes grouped_entries is a dictionary with keys as (layer, level, h_row) tuples
    """
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
            "max_mean_final_score": max_mean_final
        }

        final_results_by_layer[layer].append(result_entry)

    return final_results_by_layer



def fiter_and_aggregate_entries(data: List[Dict[str, Any]], target_layers: Optional[List[int]]) -> Dict[str, Any]:

    # Dictionary to hold lists of processed entries keyed by (layer, level, h_row)
    grouped_data = defaultdict(list)

    valid_entries_count = 0
    skipped_entries_count = 0

    # Step 1: Filter and Aggregate
    for entry in data:
        # Filter by layer
        layer = entry.get('layer')
        if target_layers and layer not in target_layers:
            continue

        # Check validity (description is not null)
        if entry.get('description') is None:
            skipped_entries_count += 1
            continue

        # Calculate means
        means = calculate_entry_means(entry)

        # Group key
        level = entry.get('level')
        h_row = entry.get('h_row')
        group_key = (layer, level, h_row)

        grouped_data[group_key].append(means)
        valid_entries_count += 1
    logger.info(
        f"Processed {valid_entries_count} valid entries. Skipped {skipped_entries_count} invalid/filtered entries.")

    return grouped_data


def save_results(results: Dict[str, Any], output_path: str):
    """
    Saves the results dictionary to a JSON file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results successfully saved to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write to output file: {e}")


def main():
    args = parse_arguments()

    logger.info(f"Starting processing. Input: {args.input_file}, Target Layers: {args.layers or 'All'}")

    raw_data = load_data(args.input_file)

    create_path_if_not_exists(args.output_file, is_file=True)

    # Get filtered and aggregated data
    grouped_data = fiter_and_aggregate_entries(raw_data, args.layers)
    final_results_by_layer = get_best_results_and_organize_by_layer(grouped_data)

    if not final_results_by_layer:
        logger.warning("No results found for the specified layers.")

    save_results(final_results_by_layer, args.output_file)
    logger.info("Processing complete.")


if __name__ == "__main__":
    setup_logging()
    main()