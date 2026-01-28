from typing import Optional

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
    ax.legend(title=hue, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

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
    ax.legend(title=hue, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig

