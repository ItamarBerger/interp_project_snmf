from enum import StrEnum
import networkx as nx
import random
import pandas as pd
import argparse
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from seaborn import color_palette

from experiments.evaluation.concept_trees.concept_tree_utils import discover_trees, parse_int_list
import logging
from utils import setup_logging

logger = logging.getLogger(__name__)


class AnalyzeTreeStrategy(StrEnum):
    RANDOM = "random"  # Randomly select one tree for each description
    AVERAGE = "average"  # Average trajectories across trees with the same description group


def get_num_of_features_at_level(top_rank: int, level: int) -> int:
    return top_rank // (2**level)


def get_level_description_data(grouped_descriptions: dict, model_name: str, layer: int, top_rank: int, root_level: int):
    model_grouped_desc = grouped_descriptions.get(model_name, {})
    layer_data = model_grouped_desc.get(f"k{top_rank}", {}).get(str(layer), [])
    level_data = None
    for data in layer_data:
        if data["level"] == root_level:
            level_data = data
            break
    return level_data

def get_group_for_description(description_grouping: dict, description: str):
    for group, descriptions in description_grouping.items():
        if description in descriptions:
            return group
    return None  # Return None if description is not found in any group

def build_snmf_dataframe(trees: list[nx.DiGraph], grouped_descriptions: dict, model_name: str, top_rank: int) -> pd.DataFrame:
    """
    Args:
        trees: A list of networkx DiGraph objects.
                   Each graph must have graph['root_node_id'].
    """
    data = []


    for tree in trees:
        # 1. Get the Root
        # We use the graph-level attribute as specified

        root_id = tree.graph["root_node_id"]
        root_node = tree.nodes[root_id]
        root_desc = root_node["description"]
        tree_id = tree.graph["tree_id"]
        root_k = tree.graph["root_k"]

        root_node_level = root_node["level"]
        root_node_layer = root_node["layer"]
        root_desc_data = get_level_description_data(grouped_descriptions, model_name=model_name, layer=root_node_layer, top_rank=top_rank, root_level=root_node_level)

        if root_desc_data is None:
            logger.warning(
                f"No grouped description data found for model {model_name}, layer {root_node_layer}, root rank {root_k}, root level {root_node_level}. Skipping tree with id {tree_id} and root description '{root_desc}'."
            )
            continue

        root_description_group = get_group_for_description(root_desc_data["grouping"], root_desc)
        if root_description_group is None:
            logger.warning(
                f"Root description '{root_desc}' of tree id {tree_id} not found in any description group for model {model_name}, layer {root_node_layer}, root rank {root_k}, root level {root_node_level}. Skipping this tree."
            )
            continue

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
                        "Tree_ID": tree_id,
                        "Root_Description": root_desc,
                        "Root_Description_Group": root_description_group,
                        # Node Specifics
                        "Node_ID": node_id,
                        "Description": node["description"],
                        "Score": node["concept_score"],
                        "Concept_Idx": node.get("concept_idx", -1),
                        "SNMF_Level": node["level"],
                        "Layer": node["layer"],
                        "Depth_Index": depth_idx,  # Implicit tree depth (0, 1, 2...)
                        "Root_K": root_k,
                    }
                    data.append(row)

    return pd.DataFrame(data)


def load_trees(trees_base_folder, root_ranks, layers=None):
    tree_paths = discover_trees(trees_base_folder, filter_k=root_ranks, layers=layers)
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
    parser.add_argument(
        "--top-rank", type=int, required=True, help="The highest rank in the SNMF decomposition (the number of features of the leaf level nodes)"
    )
    parser.add_argument(
        "--root-levels", type=parse_int_list, required=True, help="Comma-separated list of the root node level (0,1,2, etc.) used for the trees."
    )
    parser.add_argument("--layers", type=parse_int_list, required=True, help="Comma-separated list of layers to include (e.g., '0,6,12').")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[e for e in AnalyzeTreeStrategy],
        default=AnalyzeTreeStrategy.RANDOM,
        help="The strategy to use when analyzing trees with the same description group. 'random' will randomly select one tree for each description, while 'average' will average trajectories across trees with the same description group.",
    )
    return parser.parse_args()


def load_json_file(json_file):
    with open(json_file, "r") as f:
        file_contents = json.load(f)
    return file_contents


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def visualize_trajectories(
    filtered_df: pd.DataFrame, model_name: str, layer: int, root_rank: int, top_rank: int, results_base_folder: str, strategy: AnalyzeTreeStrategy
):
    """
    Visualize the trajectories of concept scores across SNMF levels for a specific layer and node rank.
    Args:
    - filtered_df: DataFrame containing the relevant data for the specified layer and node rank.
    - model_name: Name of the model (e.g., 'gpt2-small').
    - layer: The layer number being visualized.
    - root_rank: The rank of the root node (number of features).
    - results_base_folder: Directory where the resulting visualization will be saved.
    """
    plt.figure(figsize=(12, 7))

    plot_hue = "Root_Description"
    if strategy == AnalyzeTreeStrategy.AVERAGE:
        plot_hue = "Root_Description_Group"

    num_categories = filtered_df[plot_hue].nunique()
    if num_categories <= 20:
        color_palette = sns.color_palette("tab20", n_colors=num_categories)
    else:
        logger.info("Number of categories (%d) exceeds 20, using 'husl' color palette for better distinction.", num_categories)
        color_palette = sns.color_palette("husl", n_colors=num_categories)

    # 2. Draw the "Trend" (Average per Tree)
    sns.lineplot(
        data=filtered_df,
        x="SNMF_Level",
        y="Score",
        hue=plot_hue,
        estimator="mean",
        errorbar=None,  # Cleaner look without error bands
        lw=3,
        palette=color_palette
    )

    plt.title(f"{model_name} Layer {layer} Steering Score Trajectories by SNMF Level ")
    plt.xlabel("SNMF Hierarchy Level")
    plt.ylabel("Concept Score (Steering)")
    plt.grid(True, linestyle="--", alpha=1)

    # Move legend outside if you have many trees
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Root Concept")
    plt.tight_layout()

    # Force the x-axis to be integers only
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save the figure
    folder_path = os.path.join(results_base_folder, model_name, f"k{top_rank}", f"strategy_{strategy}")
    create_folder_if_not_exists(folder_path)
    save_path = os.path.join(folder_path, f"layer{layer}_rank{root_rank}_steering_trajectories.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved trajectory visualization to {save_path}")


def filter_trees_by_grouped_descriptions(
    df: pd.DataFrame,
    grouped_descriptions_data: dict,
    model_name: str,
    layer: int,
    top_rank: int,
    root_level: int,
    strategy: AnalyzeTreeStrategy = AnalyzeTreeStrategy.RANDOM,
) -> pd.DataFrame:
    """
    Filter the DataFrame to include only trees that match the specified model, layer, and root rank, and then apply the grouping strategy based on descriptions.
    Args:
        - df: The original DataFrame containing all tree data.
        - grouped_descriptions_data: A dictionary mapping model names to root ranks to layers, to lists of data containing model_name, layer, level, descriptions and grouping data
        - model_name: The name of the model to filter by (e.g., 'gpt2-small').
        - layer: The layer number to filter by.
        - root_rank: The rank of the root node (number of features) to filter by.
        - strategy: The analysis strategy to apply for trees with the same description group.
        When "random" is chosen, it chooses a description from each group randomly, and then randomly
        selects one tree that matches that description. When "average" is chosen, it averages the trajectories across all trees that match the descriptions in each group.
    """
    # Start by filtering by model, layer, and root rank
    root_rank = get_num_of_features_at_level(top_rank, root_level)
    filtered_df = df[(df["Layer"] == layer) & (df["Root_K"] == root_rank)]

    # Get the relevant grouped descriptions for this model, layer, and root rank
    level_data = get_level_description_data(grouped_descriptions_data,model_name=model_name, layer=layer, top_rank=top_rank, root_level=root_level)

    if level_data is None:
        logger.warning(
            f"No grouped descriptions found for model {model_name}, layer {layer}, root rank {root_rank}, root level {root_level}. Returning empty dataframe."
        )
        return pd.DataFrame()  # Return empty DataFrame if no descriptions found

    grouped_descriptions = level_data["grouping"]
    if strategy == AnalyzeTreeStrategy.RANDOM:
        # For each group, randomly select one description, and then randomly select one tree that matches that description
        selected_descriptions = []
        for group, descriptions in grouped_descriptions.items():
            selected_desc = random.choice(descriptions)
            selected_descriptions.append(selected_desc)

        # Filter the DataFrame to include only rows with the selected descriptions
        # Randomly selecting only one tree per description by using the Tree_ID
        final_rows = []
        for desc in selected_descriptions:
            desc_rows = filtered_df[filtered_df["Root_Description"] == desc]
            if not desc_rows.empty:
                selected_tree_id = random.choice(desc_rows["Tree_ID"].unique())
                logger.info("Selected tree id %s for description '%s'", selected_tree_id, desc)
                final_rows.append(desc_rows[desc_rows["Tree_ID"] == selected_tree_id])
        if not final_rows:
            logger.warning(
                f"No trees found matching the selected descriptions for model {model_name}, layer {layer}, root rank {root_rank}, root level {root_level}. Returning empty dataframe."
            )
            return pd.DataFrame()  # Return empty DataFrame if no trees found
        return pd.concat(final_rows)

    elif strategy == AnalyzeTreeStrategy.AVERAGE:
        # The actual averaging will be done in the visualization function by setting estimator='mean' in sns.lineplot
        # so we don't need to do any additional processing here. We just need to use the "Root_Description_Group" as the hue in the plot.
        return filtered_df

    else:
        logger.error(f"Unknown analysis strategy: {strategy}. Returning empty dataframe.")
        return pd.DataFrame()  # Return empty DataFrame for unknown strategy


def process_and_visualize(args, df: pd.DataFrame, grouped_descriptions_data: dict):
    for layer in args.layers:
        for root_level in args.root_levels:
            root_node_rank = get_num_of_features_at_level(args.top_rank, root_level)
            logger.info(f"Processing model {args.model_name}, layer {layer}, root rank {root_node_rank} (root level {root_level})")

            filtered_df = filter_trees_by_grouped_descriptions(
                df=df,
                grouped_descriptions_data=grouped_descriptions_data,
                model_name=args.model_name,
                layer=layer,
                top_rank=args.top_rank,
                root_level=root_level,
                strategy=args.strategy,
            )

            if filtered_df.empty:
                logger.warning(
                    f"No data to visualize for model {args.model_name}, layer {layer}, root rank {root_node_rank} (root level {root_level}). Skipping visualization."
                )
                continue

            visualize_trajectories(
                filtered_df=filtered_df,
                model_name=args.model_name,
                layer=layer,
                root_rank=root_node_rank,
                top_rank=args.top_rank,
                results_base_folder=args.results_base_folder,
                strategy=args.strategy,
            )


def main():
    setup_logging()
    args = parse_args()

    root_ranks = [get_num_of_features_at_level(args.top_rank, level) for level in args.root_levels] if args.root_levels else None
    # load trees
    trees = load_trees(trees_base_folder=args.trees_base_folder, root_ranks=root_ranks, layers=args.layers)
    if not trees:
        logger.error("No valid trees found. Exiting.")
        return

    # Make sure the results folder exists
    create_folder_if_not_exists(args.results_base_folder)

    # Load grouped descriptions
    grouped_descriptions = load_json_file(args.grouped_descriptions_file)

    # build dataframe
    df = build_snmf_dataframe(trees, grouped_descriptions, model_name=args.model_name, top_rank=args.top_rank)



    process_and_visualize(args=args, df=df, grouped_descriptions_data=grouped_descriptions)


if __name__ == "__main__":
    main()
