import argparse
import pickle
import os
import sys

import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)




# ------------------------------
# Utils
# ------------------------------
def log(txt: str) -> None:
    logger.info(txt)

def parse_int_list(spec: str):
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out

from utils import setup_logging
import json
import numpy as np
import csv

def main():
    """
    1. Load concept trees analaysis.
    2 Save aggregated analaysis to csv format
        2.1 - Layers Comparison
        2.2 - Levels Comparison
    3. Create Visualizations for key findings.
    """

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run causal interventions from NMF factors; auto-select Gemma vs Regular vector construction."
    )
    parser.add_argument("--layers", required=True, type=parse_int_list,
                        help="Layers to use, e.g. '0,6,12,18,25' or '0-4'.")

    parser.add_argument("--concept-trees-analysis-path", required=True, type=str,
                        help="Path to the concept trees analysis JSON file.")

    parser.add_argument("--output-path", required=True, type=str,
                        help="Base output path for concept trees analaysis figures an tables.")
    
    args = parser.parse_args()
    layers = args.layers
    concept_trees_analysis_path = args.concept_trees_analysis_path
    output_path = args.output_path


    # Create layers comparison
    rows = []
    # Columns: avg_depth, avg_leaf_ratio, % of shallow trees.
    for layer in layers:
        # 1. Load layer's concept trees analysis
        analysis_path = os.path.join(concept_trees_analysis_path, f"concept_trees_stats_layer{layer}.json")
        if not os.path.exists(analysis_path):
            log(f"Analysis file not found for layer {layer} at {analysis_path}. Skipping.")
            continue
        with open(analysis_path, 'r') as f:
            stats = json.load(f)
        

        # Levels comparison stats
        level_rows = []
        branching_dict = stats['avg_branching_per_level']
        for level, avg_level_branching in branching_dict.items():
            # rearragnge levels (json ordering is reveresed)
            level_num = len(branching_dict) - int(level) - 1
            
            # insert row
            level_rows.append({
                'layer': layer,
                'level': level_num,
                'avg_branching': float(avg_level_branching)
            })

        # save level comparison csv. each layer separately
        level_output_path = os.path.join(output_path, "concept_tree_levels_comparison")
        os.makedirs(level_output_path, exist_ok=True)
        level_csv_path = os.path.join(level_output_path, f"concept_tree_levels_comparison_{layer}.csv")
        with open(level_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "level", "avg_branching"]
            )
            writer.writeheader()
            writer.writerows( sorted(level_rows, key=lambda x: x["level"])
    )



        avg_depth = float(stats['avg_depth'])
        leaf_ratios = stats['leaf_ratio_per_tree']
        avg_leaf_ratio = float(np.mean(leaf_ratios))
        shallow_ratio = float(np.sum(np.array(leaf_ratios) == 1.0) / len(leaf_ratios))

        rows.append({
            'layer': layer,
            'avg_depth': avg_depth,
            'avg_leaf_ratio': avg_leaf_ratio,
            'shallow_tree_ratio': shallow_ratio
        })
        log(f"Loaded stats for layer {layer}")

    
    #2.1 Save aggregated analaysis to csv format
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "concept_tree_layers_comparison.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "avg_depth", "avg_leaf_ratio", "shallow_tree_ratio"]
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x["layer"]))

    log(f"Saved summary CSV to {csv_path}")





            









if __name__== '__main__':
    main()