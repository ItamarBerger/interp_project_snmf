import argparse
import os
import json
from collections import defaultdict
from typing import Optional

from experiments.evaluation.eval_utils import (
    get_best_results_and_organize_by_layer,
    filter_and_aggregate_entries,
    load_data,
)
from utils import setup_logging
from experiments.evaluation.concept_trees.concept_tree_utils import (
    parse_int_list,
    build_concept_tree,
    load_nmf_decompositions, get_node_id,
)
from enum import StrEnum
import logging
import networkx as nx

logger = logging.getLogger(__name__)


# Constants
class DescriptionType(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


INPUT_CAUSAL_RESULTS_FILE = os.path.join("{steering_base_path}", "causal_results_in.json")
OUTPUT_CAUSAL_RESULTS_FILE = os.path.join("{steering_base_path}", "causal_results_out.json")
DATA_PATH_TEMPLATE = os.path.join("{steering_base_path}", "concept_trees", "{description_type}", "extended_concept_trees_layer_{layer}.graphml")
MISSING_LAYER_WARNING_THRESHOLD = 100
FACTORIZATION_BASE_TEMPLATE = os.path.join("{steering_base_path}", "hier")
TREE_OUTPUT_PATH = os.path.join("{steering_base_path}", "concept_trees", "layer_{layer}", "concept_tree_{tree_id}.graphml")
LAYER_TREE_ID_INC = defaultdict(int)
CONCEPT_CONTEXTS_FILE = os.path.join("{steering_base_path}", "concept_contexts.json")


def get_output_path_for_tree(base_steering_path: str, layer: int, tree_id: int):
    output_file = TREE_OUTPUT_PATH.format(steering_base_path=base_steering_path, layer=layer, tree_id=tree_id)
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    return output_file


def get_tree_id_for_layer(layer: int):
    # For now, just a stupid method, might be more sophisticated later
    next_id = LAYER_TREE_ID_INC[layer]
    LAYER_TREE_ID_INC[layer] += 1
    return next_id


def build_tree(tree: nx.DiGraph, node: dict, layer: int, level: int, layer_steering_data: dict, parent_id: Optional[str] = None):
    """
    Recursively build tree as nx.DiGraph.
    """
    concept_idx = node["concept"]
    node_id = get_node_id(tree.graph["tree_id"], layer, level, concept_idx)

    node_data = layer_steering_data.get((level, concept_idx), {})
    if not node_data:
        logger.warning("Could not find node data for layer: %s level: %s concept_idx: %s", layer, level, concept_idx)

    # Get concept context if available


    top_indices = node.get("top_indices", [])

    # Add root node
    tree.add_node(
        node_id,
        layer=layer,
        level=level,
        concept_idx=concept_idx,
        concept_score=node_data["max_mean_concept_score"],
        fluency_score=node_data["max_mean_fluency_score"],
        final_score=node_data["max_mean_final_score"],
        description=node_data["description"],
        top_indices=json.dumps(top_indices),
    )

    if parent_id is not None:
        tree.add_edge(parent_id, node_id)

    for child in node.get("children", []):
        build_tree(
            tree=tree,
            node=child,
            layer=layer,
            level=child["level"],
            layer_steering_data=layer_steering_data,
            parent_id=node_id
        )


def save_trees_to_files(trees: list[nx.DiGraph], layer: int, base_steering_path: str) -> bool:
    # Create a single graph just for saving
    try:
        for tree in trees:
            tree_id = tree.graph["tree_id"]
            output_path = get_output_path_for_tree(base_steering_path=base_steering_path, layer=layer, tree_id=tree_id)
            nx.write_graphml(tree, output_path)
            logger.info("Saved tree %s for layer %s to %s", tree_id, layer, output_path)
        return True
    except Exception as e:
        logger.error("Couldn't save file", exc_info=e)
        return False


def build_trees_for_layer(
    layer: int,
    ranks: list[int],
    layer_steering_data: dict,
    nmf_decompositions,
    top_k_factors: int,
    top_k_tokens: int,
    minimal_activation: float,
) -> list[nx.DiGraph]:
    trees: list[nx.DiGraph] = []
    nmf_list = nmf_decompositions["pretrained_layers"]
    for concept_idx in range(ranks[-1]):
        level_idx = len(nmf_list) - 1
        concept_tree = build_concept_tree(
            levels=nmf_list,
            concept_idx=concept_idx,
            level_idx=level_idx,
            top_k_factors=top_k_factors,
            top_k_tokens=top_k_tokens,
            minimal_activation=minimal_activation,
        )
        tree_id = get_tree_id_for_layer(layer)
        tree = nx.DiGraph(tree_id=tree_id)

        build_tree(tree, concept_tree, layer, level_idx, layer_steering_data)
        trees.append(tree)

    return trees


def validate_args(args):
    if not args.input_layers and not args.output_layers:
        raise ValueError("At least one of --input-layers or --output-layers must be specified.")

    if args.input_layers and not args.causal_input_file:
        args.causal_input_file = INPUT_CAUSAL_RESULTS_FILE.format(steering_base_path=args.base_steering_path)
        logger.warning("Causal output file is not provided. Using default: %s", args.causal_input_file)

    if args.output_layers and not args.causal_output_file:
        args.causal_output_file = OUTPUT_CAUSAL_RESULTS_FILE.format(steering_base_path=args.base_steering_path)
        logger.warning("Causal input file is not provided. Using default: %s", args.causal_output_file)


def load_best_results_for_feature(causal_results_path: str, layers: list[int]) -> dict:
    raw_causal_results = load_data(causal_results_path)
    grouped_results, layers_with_missing = filter_and_aggregate_entries(raw_causal_results, layers)
    for layer, layer_entries in layers_with_missing:
        count = len(layer_entries)
        if count > MISSING_LAYER_WARNING_THRESHOLD:
            logger.warning("Layer %s is missing % entries. ", layer, count)
    best_results = get_best_results_and_organize_by_layer(grouped_results, use_extended_keys=True, use_dict=True)

    return best_results


def process_layers(
    args,
    description_type: DescriptionType,
):
    if description_type == DescriptionType.INPUT:
        layers = args.input_layers
        causal_results_file = args.causal_input_file
    else:
        layers = args.output_layers
        causal_results_file = args.causal_output_file
    factorization_base_path = FACTORIZATION_BASE_TEMPLATE.format(steering_base_path=args.base_steering_path)
    best_input_results = load_best_results_for_feature(causal_results_file, layers)
    nmf_decompositions = load_nmf_decompositions(layers=layers, factorization_base_path=factorization_base_path, ranks=args.ranks)

    for layer in layers:
        layer_trees = build_trees_for_layer(
            layer=layer,
            ranks=args.ranks,
            layer_steering_data=best_input_results[layer],
            nmf_decompositions=nmf_decompositions[layer],
            top_k_factors=args.top_k_factors,
            top_k_tokens=args.top_k_tokens,
            minimal_activation=args.minimal_activation,
        )
        if save_trees_to_files(layer_trees, layer, args.base_steering_path):
            logger.info("Successfully saved trees for layer %s", layer)


def load_concept_contexts(file_path: str) -> dict:
    concept_contexts = load_data(file_path)
    # Create a dict keyed by (layer, level, concept_idx) for easy lookup
    concept_contexts_dict = {}
    for entry in concept_contexts:
        key = (entry["layer"], entry["level"], entry["h_row"])
        concept_contexts_dict[key] = entry

    return concept_contexts_dict

def main():
    parser = argparse.ArgumentParser(
        description="Build concept trees for a given model. Assumes the following structure under base path: base -> <some-results>.json hier/ ->"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g., 'gpt2-small').")
    parser.add_argument(
        "--base-steering-path",
        type=str,
        required=True,
        help="Base path where factorization and steering results are stored.",
    )
    parser.add_argument(
        "--ranks",
        type=parse_int_list,
        required=True,
        help="A comma-separated list of NMF ranks to consider (e.g., '200,100,50').",
    )
    parser.add_argument(
        "--input-layers",
        type=parse_int_list,
        help="A comma-separated list of  layer to consider for input descriptions.",
    )
    parser.add_argument(
        "--output-layers",
        type=parse_int_list,
        help="A comma-separated list of  layer to consider for output descriptions.",
    )
    parser.add_argument(
        "--causal-input-file",
        type=str,
        default=None,
        help="Path to the JSON file containing the causal results relating to input descriptions.",
    )
    parser.add_argument(
        "--causal-output-file",
        type=str,
        default=None,
        help="Path to the JSON file containing the causal results relating to output descriptions.",
    )
    parser.add_argument("--top-k-factors", type=int, default=5, help="Number of top factors to include at each level of the tree.")
    parser.add_argument("--top-k-tokens", type=int, default=10, help="Number of top tokens to include for each concept in the tree.")
    parser.add_argument(
        "--minimal-activation",
        type=float,
        default=0.1,
        help="Minimum activation threshold (as a fraction of max) for including child nodes in the tree.",
    )

    args = parser.parse_args()

    validate_args(args)

    logger.info("===== Starting concept tree construction with the following parameters =====")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Base Steering Path: {args.base_steering_path}")
    logger.info(f"Ranks: {args.ranks}")
    logger.info(f"Input Layers: {args.input_layers} with descriptions from {args.causal_input_file}")
    logger.info(f"Output Layers: {args.output_layers} with descriptions from {args.causal_output_file}")
    logger.info("===========================================================")

    if args.input_layers:
        process_layers(args, DescriptionType.INPUT)

    if args.output_layers:
        process_layers(args, DescriptionType.OUTPUT)


if __name__ == "__main__":
    setup_logging()
    main()
