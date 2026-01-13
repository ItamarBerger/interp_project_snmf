import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ------------------------------
# Paths
# ------------------------------
LEVEL_CSV_DIR = "./concept_tree_levels_comparison"  # folder containing level comparison of specific layer CSVs
OUTPUT_FILE = "./branching_across_layers.png"

# ------------------------------
# Load all CSVs (each corresponds to a layer)
# ------------------------------
all_files = sorted(glob.glob(os.path.join(LEVEL_CSV_DIR, "concept_tree_levels_comparison_*.csv")))
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    # Exclude only level 0
    df = df[df["level"] != 0]
    df_list.append(df)

branching_df = pd.concat(df_list, ignore_index=True)

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(8, 5))

layers = sorted(branching_df["layer"].unique())
levels = sorted(branching_df["level"].unique())

# Generate distinct colors for levels
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(levels))]

bar_width = 0.2
x = range(len(layers))  # x positions for layers

for i, level in enumerate(levels):
    level_data = branching_df[branching_df["level"] == level].sort_values("layer")
    y = level_data["avg_branching"].values
    # offset bars to avoid overlap
    plt.bar([xi + i*bar_width for xi in x], y, width=bar_width, color=colors[i], label=f"Level {level}")

# Adjust x-axis
plt.xticks([xi + bar_width*(len(levels)-1)/2 for xi in x], layers)
plt.xlabel("Model layer")
plt.ylabel("Average branching factor")
plt.title("Branching Factor per Layer and Level (excluding level 0)")

plt.legend(title="Level", frameon=True)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()

# Save figure
plt.savefig(OUTPUT_FILE, dpi=150)
plt.close()
print(f"Saved figure to {OUTPUT_FILE}")
