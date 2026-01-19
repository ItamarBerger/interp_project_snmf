import argparse
import logging
import sys
import pandas as pd
from enum import StrEnum
from typing import Dict, Any

from experiments.evaluation.eval_utils import load_data, create_path_if_not_exists
from experiments.evaluation.vis_utils import save_plt, plot_boxplot, plot_barplot
from utils import setup_logging

logger = logging.getLogger(__name__)


class PlotType(StrEnum):
    LAYER_BOXPLOT = "layer_boxplot"
    LEVEL_BOXPLOT = "level_boxplot"
    MEAN_LAYER_BARPLOT = "mean_layer_barplot"
    MEAN_LEVEL_BARPLOT = "mean_level_barplot"

    def get_graph_type(self):
        parts = self.value.spit("_")
        return parts[-1]


class Metric(StrEnum):
    CONCEPT_SCORE = "concept"
    FINAL_SCORE = "final"
    FLUENCY_SCORE = "fluency"

    def get_metric_df_name(self):
        mapping = {
            Metric.CONCEPT_SCORE: "Concept Score",
            Metric.FINAL_SCORE: "Final Score",
            Metric.FLUENCY_SCORE: "Fluency Score"
        }
        return mapping[self]


class PlotMode(StrEnum):
    SINGLE_PLOT = "single_plot"
    MULTI_PLOT = "multi_plot"


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

    parser.add_argument("--layers", type=str, nargs='*', default=None,
                        help="Specific layers to visualize. If not set, all layers are visualized.")

    parser.add_argument("--metrics", type=str, nargs='*',
                        default=None, help="Filter by specific metrics (concept, final, fluency).")

    parser.add_argument("--mode", type=str, choices=[e.value for e in PlotMode], default=PlotMode.MULTI_PLOT,
                        help="Plotting mode: single_plot or multi_plot.")

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


def plot_layer_distribution(df: pd.DataFrame, output_prefix: str, mode: PlotMode = PlotMode.MULTI_PLOT):
    """
    Generates and saves a boxplot for a specific layer across all given levels.
    """
    if mode == PlotMode.MULTI_PLOT:
        for layer_id in df['Layer'].unique():
            logger.info(f"Processing layer {layer_id}...")
            layer_df = df[df['Layer'] == layer_id]
            if layer_df.empty:
                logger.warning(f"No data found for Layer {layer_id}. Skipping plot.")
                return

            fig = plot_boxplot(layer_df, f"Score Distribution - Layer {layer_id}", "Level", "Score", "Metric",
                         (MIN_SCORE - BUFFER, MAX_SCORE + BUFFER))

            # Construct output filename
            output_path = f"{output_prefix}_layer_{layer_id}.png"

            save_plt(fig, f"score distribution plot for Layer {layer_id}", output_path, logger)
    else:
        logger.info("Plotting all layers in one plot per metric...")
        metrics = df['Metric'].unique()
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            fig = plot_boxplot(
                df=metric_df,
                plt_title=f"Score Distribution - Metric: {metric}",
                x="Level",
                y="Score",
                hue="Layer",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER)
            )

            output_path = f"{output_prefix}_all_layers_metric_{metric.lower().replace(' ', '_')}.png"
            save_plt(fig, f"score distribution plot for Metric {metric}", output_path, logger)


def plot_level_distribution(df: pd.DataFrame, output_prefix: str, mode: PlotMode = PlotMode.MULTI_PLOT):
    """
    Generates and saves a boxplot for a specific level across all given layers.
    """
    if mode == PlotMode.MULTI_PLOT:
        for level in df['Level'].unique():
            logger.info(f"Processing level {level}...")
            level_df = df[df['Level'] == level]
            if level_df.empty:
                logger.warning(f"No data found for Level {level}. Skipping plot.")
                return

            fig = plot_boxplot(
                df=level_df,
                plt_title=f"Score Distribution - Level {level}",
                x="Layer",
                y="Score",
                hue="Metric",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER))

            # Construct output filename
            output_path = f"{output_prefix}_level_{level}.png"

            save_plt(fig, f"score distribution plot for Level {level}", output_path, logger)
    else:
        logger.info("Plotting all levels in one plot per metric...")
        metrics = df['Metric'].unique()
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            fig = plot_boxplot(
                df=metric_df,
                plt_title=f"Score Distribution - Metric: {metric}",
                x="Layer",
                y="Score",
                hue="Level",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER)
            )

            output_path = f"{output_prefix}_all_levels_metric_{metric.lower().replace(' ', '_')}.png"
            save_plt(fig, f"score distribution plot for Metric {metric}", output_path, logger)


def plot_level_means_for_layer(df: pd.DataFrame, output_prefix: str, mode: PlotMode = PlotMode.MULTI_PLOT):
    """
    Generates and saves a bar plot of mean scores for a specific layer.
    """
    if mode == PlotMode.MULTI_PLOT:
        for layer_id in df['Layer'].unique():
            layer_df = df[df['Layer'] == layer_id]
            logger.info(f"Processing layer {layer_id}...")
            if layer_df.empty:
                logger.warning(f"No data found for Layer {layer_id}. Skipping plot.")
                return

            fig = plot_barplot(
                df=layer_df,
                plt_title=f"Mean Scores - Layer {layer_id}",
                x="Level",
                y="Score",
                hue="Metric",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER),
                y_label="Mean Score"
            )
            # Construct output filename
            output_path = f"{output_prefix}_mean_layer_{layer_id}.png"

            save_plt(fig, f"mean score plot for Layer {layer_id}", output_path, logger)
    else:
        logger.info("Plotting all layers in one plot per metric...")
        metrics = df['Metric'].unique()
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            fig = plot_barplot(
                df=metric_df,
                plt_title=f"Mean Scores - Metric: {metric}",
                x="Level",
                y="Score",
                hue="Layer",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER),
                y_label="Mean Score",
                bar_label_fontsize=8
            )

            layers_string = "_".join(str(layer) for layer in sorted(df['Layer'].unique()))
            output_path = f"{output_prefix}_mean_level_layers_{layers_string}_metric_{metric.lower().replace(' ', '_')}.png"
            save_plt(fig, f"mean score plot for Metric {metric}", output_path, logger)


def plot_layer_means_for_level(df: pd.DataFrame, output_prefix: str, mode: PlotMode = PlotMode.MULTI_PLOT):
    """
    Generates and saves a bar plot of mean scores for a specific level across all given layers.
    arguments:
    - df: the relevant data frame
    - level: The level identifier
    - output_prefix: Prefix for the output file name
    """
    if mode == PlotMode.MULTI_PLOT:
        for level in df['Level'].unique():
            logger.info(f"Processing level {level}...")
            level_df = df[df['Level'] == level]
            if level_df.empty:
                logger.warning(f"No data found for Level {level}. Skipping plot.")
                return

            fig = plot_barplot(
                df=level_df,
                plt_title=f"Mean Scores - Level {level}",
                x="Layer",
                y="Score",
                hue="Metric",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER),
                y_label="Mean Score"
            )
            # Construct output filename
            output_path = f"{output_prefix}_mean_level_{level}.png"

            save_plt(fig, f"mean score plot for Level {level}", output_path, logger)
    else:
        logger.info("Plotting all levels in one plot per metric...")
        metrics = df['Metric'].unique()
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            fig = plot_barplot(
                df=metric_df,
                plt_title=f"Mean Scores - Metric: {metric}",
                x="Layer",
                y="Score",
                hue="Level",
                ylim=(MIN_SCORE - BUFFER, MAX_SCORE + BUFFER),
                y_label="Mean Score",
                bar_label_fontsize=8,
            )

            layers_string = "_".join(str(layer) for layer in sorted(df['Layer'].unique()))
            output_path = f"{output_prefix}_mean_layer_layers_{layers_string}_metric_{metric.lower().replace(' ', '_')}.png"
            save_plt(fig, f"mean score plot for Metric {metric}", output_path, logger)


def main():
    args = parse_arguments()

    logger.info("Starting visualization...")

    data = load_data(args.input_file)

    # create output directory if it doesn't exist
    create_path_if_not_exists(args.output_prefix)

    df = transform_data_to_df(data, args.include_fluency)

    if args.metrics is not None:
        # Filter the dataframe if metric is given
        metric_filter = [Metric(m).get_metric_df_name() for m in args.metrics]
        df = df[df['Metric'].isin(metric_filter)]

    if args.layers is not None:
        # Use only specified layers
        layers = [int(layer) for layer in args.layers if int(layer) in df['Layer'].unique()]
        df = df[df['Layer'].isin(layers)]

    plot_to_func = {
        PlotType.LAYER_BOXPLOT: plot_layer_distribution,
        PlotType.LEVEL_BOXPLOT: plot_level_distribution,
        PlotType.MEAN_LAYER_BARPLOT: plot_level_means_for_layer,
        PlotType.MEAN_LEVEL_BARPLOT: plot_layer_means_for_level,
    }

    if args.graph_type in plot_to_func:
        plot_func = plot_to_func[args.graph_type]
        plot_func(df=df, output_prefix=args.output_prefix, mode=PlotMode(args.mode))

    else:
        logger.error(f"Graph type {args.graph_type} not implemented.")

    logger.info("Visualization complete.")


if __name__ == "__main__":
    setup_logging()
    main()
