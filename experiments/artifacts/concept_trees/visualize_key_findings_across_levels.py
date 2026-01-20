import pandas as pd
import matplotlib.pyplot as plt
import os
import glob



# -------------------------------
# Model
# -------------------------------
MODEL_NAME = "gemma_2_2b"

# ------------------------------
# Paths
# ------------------------------
LEVEL_CSV_DIR = f"experiments/artifacts/concept_trees/{MODEL_NAME}_analysis/concept_trees_analysis_visualizations/concept_tree_layers_comparison"
OUTPUT_DIR = f"experiments/artifacts/concept_trees/{MODEL_NAME}_analysis/concept_trees_analysis_visualizations"

OUTPUT_FILE_ALL = os.path.join(OUTPUT_DIR, "branching_across_layers.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load all CSVs (each corresponds to a layer)
# ------------------------------
all_files = sorted(glob.glob(os.path.join(
    LEVEL_CSV_DIR, "concept_tree_levels_comparison_*.csv"
)))

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df = df[df["level"] != 0]  # exclude level 0
    df_list.append(df)

branching_df = pd.concat(df_list, ignore_index=True)

layers = sorted(branching_df["layer"].unique())
levels = sorted(branching_df["level"].unique())

# ------------------------------
# Color mapping (GLOBAL & CONSISTENT)
# ------------------------------
cmap = plt.get_cmap("tab10")
level_to_color = {
    level: cmap(i) for i, level in enumerate(levels)
}

# ------------------------------
# FIGURE 1: All levels together
# ------------------------------
plt.figure(figsize=(8, 5))

bar_width = 0.2
x = range(len(layers))

for i, level in enumerate(levels):
    level_data = (
        branching_df[branching_df["level"] == level]
        .sort_values("layer")
    )
    y = level_data["avg_branching"].values

    plt.bar(
        [xi + i * bar_width for xi in x],
        y,
        width=bar_width,
        color=level_to_color[level],
        label=f"Level {level}",
    )

plt.xticks(
    [xi + bar_width * (len(levels) - 1) / 2 for xi in x],
    layers
)
plt.xlabel("Model layer")
plt.ylabel("Average branching factor")
plt.title("Branching Factor per Layer and Level (excluding level 0)")
plt.legend(title="Level", frameon=True)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()

plt.savefig(OUTPUT_FILE_ALL, dpi=150)
plt.close()
print(f"Saved figure to {OUTPUT_FILE_ALL}")

# ------------------------------
# FIGURES 2+: One figure per level (creating less cluttered visualizations)
# ------------------------------
for level in levels:
    plt.figure(figsize=(7, 4))

    level_data = (
        branching_df[branching_df["level"] == level]
        .sort_values("layer")
    )

    plt.bar(
        level_data["layer"],
        level_data["avg_branching"],
        color=level_to_color[level],
        width=0.6,
    )

    plt.xlabel("Model layer")
    plt.ylabel("Average branching factor")
    plt.title(f"Branching Factor Across Layers — Level {level}")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR, f"branching_across_layers_level_{level}.png"
    )
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved figure to {output_file}")
