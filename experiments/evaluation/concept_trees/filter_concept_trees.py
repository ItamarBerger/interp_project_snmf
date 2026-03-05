#!/usr/bin/env python3
"""
Filter concept trees and compute concept score differences between levels.

This script processes concept trees stored as graphml files and computes
the average concept score differences between each pair of levels in the tree.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import logging
from utils import setup_logging
from enum import StrEnum
import networkx as nx
from experiments.evaluation.concept_trees.concept_tree_utils import discover_trees

logger = logging.getLogger(__name__)


class ScoreType(StrEnum):
    CONCEPT_SCORE = "concept_score"
    FLUENCY_SCORE = "fluency_score"
    FINAL_SCORE = "final_score"


def load_tree(graphml_path: str) -> nx.DiGraph:
    """Load a concept tree from a graphml file."""
    return nx.read_graphml(graphml_path)


def get_tree_metadata(tree: nx.DiGraph) -> Dict[str, Any]:
    """Extract tree_id and root_k from the tree's graph attributes."""
    tree_id = tree.graph["tree_id"]
    root_k = tree.graph["root_k"]
    return {"tree_id": tree_id, "root_k": int(root_k), "root_node_id": tree.graph["root_node_id"]}


def get_node_metadata(tree: nx.DiGraph, node_id: str) -> Dict[str, Any]:
    """Extract metadata for a specific node."""
    if node_id not in tree.nodes:
        raise ValueError(f"Node ID '{node_id}' not found in the tree.")
    node_data = tree.nodes[node_id]
    return {
        "layer": int(node_data["layer"]),
        "level": int(node_data["level"]),
        "concept_idx": int(node_data["concept_idx"]),
        "concept_score": float(node_data["concept_score"])
    }

def compute_average_scores_by_level(tree: nx.DiGraph, score_type: ScoreType) -> Dict[int, float]:
    """
    Compute the average  score for each level in the tree.

    Returns:
        Dictionary mapping level -> average concept score
    """
    level_scores = defaultdict(list)

    for node, data in tree.nodes(data=True):
        level = int(data.get("level"))
        score = float(data.get(score_type))
        level_scores[level].append(score)

    avg_scores = {}
    for level, scores in level_scores.items():
        avg_scores[level] = sum(scores) / len(scores)

    return avg_scores


def compute_score_diffs(avg_scores: Dict[int, float]) -> Dict[str, float]:
    """
    Compute the differences in average concept scores between consecutive levels.

    The comparisons are made from higher level to lower level, including:
    - Root to each lower level
    - Each adjacent pair of levels

    For example, with levels [3, 2, 1, 0]:
    Computes: 3->2, 2->1, 1->0 (consecutive pairs, higher to lower)

    Returns:
        Dictionary mapping "higher_level->lower_level" -> score difference
    """
    if not avg_scores:
        return {}

    levels = sorted(avg_scores.keys(), reverse=True)  # Sort descending (higher first)

    diffs = {}

    # Compute differences between consecutive levels (higher to lower)
    for i in range(len(levels) - 1):
        higher_level = levels[i]
        lower_level = levels[i + 1]
        key = f"{higher_level}->{lower_level}"
        diffs[key] = avg_scores[higher_level] - avg_scores[lower_level]

    return diffs


def process_tree(graphml_path: str) -> Dict[str, Any] | None:
    """
    Process a single concept tree file.

    Returns:
        Dictionary with tree_id, root_k, and concept_score_diffs, or None if single-node tree.
    """
    tree = load_tree(graphml_path)

    # Filter out single-node trees
    if tree.number_of_nodes() <= 1:
        return None

    metadata = get_tree_metadata(tree)
    # Get the data of the root node
    root_node_data = get_node_metadata(tree, metadata["root_node_id"])
    avg_concept_scores = compute_average_scores_by_level(tree, ScoreType.CONCEPT_SCORE)
    score_diffs = compute_score_diffs(avg_concept_scores)

    tree_branching_factor = sum(tree.out_degree(node) for node in tree.nodes()) / tree.number_of_nodes()

    tree_data =  {
        "tree_id": metadata["tree_id"],
        "root_k": metadata["root_k"],
        "layer":  root_node_data["layer"],
        "branching_factor": tree_branching_factor,
        "avg_concept_scores_by_level": avg_concept_scores,
        "avg_fluency_scores_by_level": compute_average_scores_by_level(tree, ScoreType.FLUENCY_SCORE),
        "avg_final_scores_by_level": compute_average_scores_by_level(tree, ScoreType.FINAL_SCORE),
        "concept_score_diffs": score_diffs,
        "number_of_nodes": tree.number_of_nodes()
    }

    # Add the number of children at each level (for additional analysis)
    level_children_counts = defaultdict(int)
    for node, data in tree.nodes(data=True):
        level = int(data.get("level"))
        level_children_counts[level] += tree.out_degree(node)
    tree_data["level_children_counts"] = dict(level_children_counts)

    # Add the actual depth of the tree
    tree_data["tree_depth"] = max(avg_concept_scores.keys()) - min(avg_concept_scores.keys())
    return tree_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter concept trees and compute concept score differences between levels."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the directory containing concept trees. Expected structure: K{rank}/layer_{layer}/*.graphml"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="filtered_concept_trees.json",
        help="Output JSON file path (default: filtered_concept_trees.json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.isdir(args.input_path):
        logger.error(f"Error: Input path '{args.input_path}' is not a valid directory.")
        return 1

    # Discover all tree files
    graphml_files = discover_trees(args.input_path)

    if not graphml_files:
        logger.info(f"No graphml files found under '{args.input_path}'")
        return 1

    if args.verbose:
        logger.info(f"Found {len(graphml_files)} graphml files")

    # Process all trees
    results = []
    single_node_count = 0

    for graphml_path in graphml_files:
        result = process_tree(graphml_path)

        if result is None:
            single_node_count += 1
            if args.verbose:
                logger.info(f"Skipped single-node tree: {graphml_path}")
        else:
            results.append(result)

    # Print statistics
    logger.info(f"Processed {len(graphml_files)} trees")
    logger.info(f"  - Valid trees (multi-node): {len(results)}")
    logger.info(f"  - Single-node trees (filtered out): {single_node_count}")

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    setup_logging()
    exit(main())


