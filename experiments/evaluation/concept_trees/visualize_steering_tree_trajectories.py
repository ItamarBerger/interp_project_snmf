import networkx as nx
import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from experiments.evaluation.concept_trees.concept_tree_utils import discover_trees, parse_int_list
import logging
from utils import setup_logging

logger = logging.getLogger(__name__)

def build_snmf_dataframe(trees: list[nx.DiGraph]) -> pd.DataFrame:
    """
    Args:
        trees: A list of networkx DiGraph objects.
                   Each graph must have graph['root_node_id'].
    """
    data = []

    for tree in trees:
        # 1. Get the Root
        # We use the graph-level attribute as specified

        root_id = tree.graph['root_node_id']
        root_desc = tree.nodes[root_id]["description"]
        tree_id = tree.graph["tree_id"]
        root_k = tree.graph["root_k"]


        # Find all paths from Root to Leaves
        # Identify leaves (nodes with out-degree 0)
        leaves = [n for n in tree.nodes() if tree.out_degree(n) == 0]

        for leaf in leaves:
            # Get all simple paths
            paths = nx.all_simple_paths(tree, source=root_id, target=leaf)

            for path_idx, path in enumerate(paths):
                for depth_idx, node_id in enumerate(path):
                    node = tree.nodes[node_id]

                    # 3. Extract ALL requested attributes
                    # We use .get() to avoid crashing if data is missing
                    row = {
                        # Identifier for the Tree
                        'Tree_ID': tree_id,
                        "Root_Description": root_desc,

                        # Node Specifics
                        'Node_ID': node_id,
                        'Description': node["description"],
                        'Score': node["concept_score"],
                        'Concept_Idx': node.get('concept_idx', -1),
                        'SNMF_Level': node.get["level"],

                        'Layer': node["layer"],
                        'Depth_Index': depth_idx, # Implicit tree depth (0, 1, 2...)
                        'Root_K': root_k
                    }
                    data.append(row)

    return pd.DataFrame(data)


def load_trees(trees_base_folder, layers=None):
    tree_paths = discover_trees(trees_base_folder, layers)
    trees = []
    for path in tree_paths:
        try:
            tree = nx.read_graphml(path)

            if tree.number_of_nodes() <= 1:
                logger.info(f"Skipping tree at {path} because it has {tree.number_of_nodes()} node(s).")
                continue

            trees.append(tree)
        except Exception as e:
            logger.error(f"Error loading tree from {path}: {e}")
    return trees


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SNMF Concept Tree Trajectories")
    parser.add_argument("--trees-base-folder", type=str, required=True, help="Directory containing concept tree graphml files")
    parser.add_argument("--results-base-folder", type=str, required=True, help="Directory to save the resulting visualizations")
    parser.add_argument("--grouped-descriptions-file", type=str, required=True, help="File containing the descriptions at each layer for each model")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g., 'gpt2-small')")
    parser.add_argument("--layers", type=parse_int_list, default=None, help="Comma-separated list of layers to include (e.g., '0,6,12'). If not specified, includes all layers that it finds under tree folder")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # load trees
    trees = load_trees(args.trees_base_folder, args.layers)
    if not trees:
        logger.error("No valid trees found. Exiting.")
        return

    # build dataframe
    df = build_snmf_dataframe(trees)
