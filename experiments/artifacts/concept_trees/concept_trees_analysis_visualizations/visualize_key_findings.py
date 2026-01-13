import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "concept_tree_layers_comparison.csv"
OUTPUT_DIR = "."


def plot_shallow_ratio(df):
    layers = df["layer"]
    # Convert ratio to percentage of shallow trees
    shallow_percent = 100 * df["shallow_tree_ratio"]

    plt.figure(figsize=(6, 4))
    plt.plot(
        layers,
        shallow_percent,
        marker="o",
        color="tab:orange",
        linewidth=2
    )

    plt.xlabel("Model layer")
    plt.ylabel("(%) Trees with depth = 0")
    plt.title("Percentage of Shallow Concept Trees")
    plt.ylim(0, 100)  # percentage scale
    plt.grid(alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "shallow_tree_ratio_per_layer.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_avg_depth(df):
    layers = df["layer"]
    avg_depth = df["avg_depth"]

    plt.figure(figsize=(6, 4))
    plt.plot(
        layers,
        avg_depth,
        marker="s",
        linestyle="--",
        color="tab:blue",
        linewidth=2
    )

    plt.xlabel("Model layer")
    plt.ylabel("Average tree depth")
    plt.title("Average Concept Tree Depth Across Layers")
    plt.grid(alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "avg_tree_depth_per_layer.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH).sort_values("layer")

    plot_shallow_ratio(df)
    plot_avg_depth(df)


if __name__ == "__main__":
    main()
