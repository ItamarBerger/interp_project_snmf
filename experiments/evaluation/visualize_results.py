import argparse
import logging
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import StrEnum
from typing import Dict, Any

from experiments.evaluation.eval_utils import load_data, create_path_if_not_exists
from utils import setup_logging

logger = logging.getLogger(__name__)

class PlotType(StrEnum):
    LAYER_BOXPLOT = "layer_boxplot"
    LEVEL_BOXPLOT = "level_boxplot"
    MEAN_LAYER_BARPLOT = "mean_layer_barplot"

    def get_graph_type(self):
        parts = self.value.spit("_")
        return parts[-1]

MIN_SCORE = 0.0
MAX_SCORE = 2.0
BUFFER = 0.05


def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize experiment results per layer.")

    parser.add_argument(
        '--input-file',
        type=str,
        help="Path to the input JSON file containing processed results."
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        help="Prefix for the output image files (e.g., 'layer_plot'). Files will be named '{prefix}_layer_{id}.png'."
    )

    parser.add_argument(
        '--graph-type',
        type=str,
        choices=[e.value for e in PlotType],
        default=PlotType.LAYER_BOXPLOT,
        help="Type of graph to generate."
    )

    parser.add_argument(
        '--include-fluency',
        action='store_true',
        help="If set, includes fluency scores in the visualization."
    )

    return parser.parse_args()



def transform_data_to_df(data: Dict[str, Any], include_fluency: bool) -> pd.DataFrame:
    """
    Transforms the hierarchical dictionary data into a flat Pandas DataFrame suitable for plotting.
    """
    records = []

    # Data is expected to be Dict[layer_str, List[entry_dict]]
    for layer_key, entries in data.items():
        try:
            layer = int(layer_key)
        except ValueError:
            logger.warning(f"Skipping key '{layer_key}' as it is not a valid integer layer.")
            continue

        for entry in entries:
            level = entry.get('level')

            # Map the metric names to cleaner labels for the plot
            metrics = {
                "Concept Score": entry.get('max_mean_concept_score'),
                "Final Score": entry.get('max_mean_final_score')
            }

            if include_fluency:
                metrics["Fluency Score"] = entry.get('max_mean_fluency_score')

            # Create a row for each metric
            for metric_name, score in metrics.items():
                if score is not None:
                    records.append({
                        "Layer": layer,
                        "Level": level,
                        "Metric": metric_name,
                        "Score": score
                    })

    if not records:
        logger.error("No valid records found after processing. Check input data format.")
        sys.exit(1)

    return pd.DataFrame(records)


def plot_layer_distribution(df: pd.DataFrame, layer_id: str, output_prefix: str):
    """
    Generates and saves a boxplot for a specific layer.
    """
    if df.empty:
        logger.warning(f"No data found for Layer {layer_id}. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))

    # Generate Boxplot
    # X-axis: Rank (Level)
    # Y-axis: Score
    # Distribution is formed by the different 'h_row' values aggregated in the previous step
    sns.boxplot(
        data=df,
        x="Level",
        y="Score",
        hue="Metric",
        palette="Set2",
    )

    plt.title(f"Score Distribution - Layer {layer_id}")
    plt.xlabel("Level")
    plt.ylabel("Score")
    plt.ylim(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER)
    plt.legend(title="Metric")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Construct output filename
    output_path = f"{output_prefix}_layer_{layer_id}.png"

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot for Layer {layer_id} to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot for Layer {layer_id}: {e}")



def plot_level_distribution(df: pd.DataFrame, level: str, output_prefix: str):
    """
    Generates and saves a boxplot for a specific level across all given layers.
    """
    if df.empty:
        logger.warning(f"No data found for Level {level}. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))

    # Generate Boxplot
    sns.boxplot(
        data=df,
        x="Layer",
        y="Score",
        hue="Metric",
        palette="Set2",
    )

    plt.title(f"Score Distribution - Level {level}")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.ylim(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER)
    plt.legend(title="Metric")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Construct output filename
    output_path = f"{output_prefix}_level_{level}.png"

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot for Level {level} to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot for Level {level}: {e}")


def plot_level_means_for_layer(layer_df: pd.DataFrame, layer_id: str, output_prefix: str):
    """
    Generates and saves a bar plot of mean scores for a specific layer.
    """
    if layer_df.empty:
        logger.warning(f"No data found for Layer {layer_id}. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))


    # Generate Barplot
    barplot = sns.barplot(
        data=layer_df,
        x="Level",
        y="Score",
        hue="Metric",
        palette="Set2",
        errorbar='sd',
    )

    for container in barplot.containers:
        barplot.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)

    plt.title(f"Mean Scores - Layer {layer_id}")
    plt.xlabel("Level")
    plt.ylabel("Mean Score")
    plt.ylim(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER)
    plt.legend(title="Metric")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Construct output filename
    output_path = f"{output_prefix}_mean_layer_{layer_id}.png"

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved mean score plot for Layer {layer_id} to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save mean score plot for Layer {layer_id}: {e}")

def main():
    args = parse_arguments()

    logger.info("Starting visualization...")

    data = load_data(args.input_file)

    # create output directory if it doesn't exist
    create_path_if_not_exists(args.output_prefix)

    df = transform_data_to_df(data, args.include_fluency)

    if args.graph_type == PlotType.LAYER_BOXPLOT:
        for layer_id, entries in data.items():
            logger.info(f"Processing Layer {layer_id}...")

            layer_df = df[df['Layer'] == int(layer_id)]
            plot_layer_distribution(layer_df, layer_id, args.output_prefix)
    elif args.graph_type == PlotType.LEVEL_BOXPLOT:
        levels = df['Level'].unique()
        for level in levels:
            logger.info(f"Processing Level {level}...")

            level_df = df[df['Level'] == level]
            plot_level_distribution(level_df, level, args.output_prefix)

    elif args.graph_type == PlotType.MEAN_LAYER_BARPLOT:
        for layer_id, entries in data.items():
            logger.info(f"Processing Mean Scores for Layer {layer_id}...")

            layer_df = df[df['Layer'] == int(layer_id)]
            plot_level_means_for_layer(layer_df, layer_id, args.output_prefix)
    else:
        logger.error(f"Graph type {args.graph_type} not implemented.")



    logger.info("Visualization complete.")


if __name__ == "__main__":
    setup_logging()
    main()