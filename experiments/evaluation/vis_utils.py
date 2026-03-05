from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_plt(fig: plt.Figure, plot_name, output_path, logger):
    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved {plot_name} to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save {plot_name}: {e}")


def plot_boxplot(df, plt_title, x: str, y: str, hue: str, ylim: tuple, color_palette = "Set2") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=color_palette,
        ax=ax  # Make seaborn use the provided ax
    )

    ax.set_title(plt_title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_ylim(*ylim)
    # Move legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=hue)
    fig.tight_layout()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig

def plot_multi_boxplot(df_map: dict[str, pd.DataFrame], x: str, y: str, hue: str, ylim: tuple, color_palette = "Set2") -> plt.Figure:
    num_plots = len(df_map)
    width = 4 * num_plots

    sns.set_style("ticks")
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(width, 4), sharey=True)

    for ax, (title, df) in zip(axes, df_map.items()):
        sns.boxplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=color_palette,
            ax=ax,
        )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove legends from individual plots
    for ax in axes:
        ax.get_legend().remove()

    # Create joint legend
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)] # Flatten the lines_labels list
    unique_labels_dict = dict(zip(labels, handles))
    sorted_keys = sorted(unique_labels_dict.keys(), key=lambda x: int(x))
    final_handles = [unique_labels_dict[k] for k in sorted_keys]
    final_labels = sorted_keys

    fig.legend(
        final_handles,
        final_labels,
        title="Layer",
        loc="upper left",
        bbox_to_anchor=(1.05, 1)
    )
    fig.tight_layout()

    return fig


def plot_barplot(df, plt_title, x: str, y: str, hue: str, ylim: tuple, y_label: Optional[str] = None, bar_label_fontsize = 9, color_palette = "Set2") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=color_palette,
        errorbar='sd',
        ax=ax,  # Make seaborn use the provided ax
        err_kws={'linewidth': 0.8}
    )

    for container in barplot.containers:
        barplot.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=bar_label_fontsize)

    ax.set_title(plt_title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label if y_label else y)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=hue)
    fig.tight_layout()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig

