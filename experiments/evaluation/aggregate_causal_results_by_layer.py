import argparse
import json
import logging
import sys
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional

from experiments.evaluation.eval_utils import load_data, create_path_if_not_exists, \
    filter_and_aggregate_entries, get_best_results_and_organize_by_layer
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
    grouped_data,_ = filter_and_aggregate_entries(raw_data, args.layers)
    final_results_by_layer = get_best_results_and_organize_by_layer(grouped_data)

    if not final_results_by_layer:
        logger.warning("No results found for the specified layers.")

    save_results(final_results_by_layer, args.output_file)
    logger.info("Processing complete.")


if __name__ == "__main__":
    setup_logging()
    main()