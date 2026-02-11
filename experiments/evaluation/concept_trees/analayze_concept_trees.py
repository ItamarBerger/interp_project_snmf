import argparse
import os
import sys
import torch
import numpy as np
import networkx as nx
import json
from collections import Counter, defaultdict
from experiments.evaluation.concept_trees.concept_tree_utils import build_concept_tree, parse_int_list, load_nmf_decompositions
from utils import setup_logging
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)


# ------------------------------
# Utils
# ------------------------------
def log(txt: str) -> None:
    logger.info(txt)



def print_logit_diff(model, logits_before, logits_after):
    # Select the logits for the last token in the sequence (shape: [vocab_size])
    logits_before_last = logits_before[0, -1, :]
    logits_after_last = logits_after[0, -1, :]

    # Compute the difference in logits (after - before)
    delta_logits = (logits_after_last - logits_before_last)

    # Get the top 10 tokens with the largest positive increase
    top_increases, top_indices = torch.topk(abs(delta_logits), k=20)

    print("Tokens with the highest logit increase after intervention:")
    for token_id, increase in zip(top_indices, top_increases):
        # Convert token ID to string using your model's tokenizer
        # Here we assume feature_processor._model.to_str_tokens returns a readable token string
        token_str = model.to_str_tokens([token_id])
        print(f"Token: {token_str}, Increase: {increase.item():.4f}")


def print_concept_tree(node, token_ds, indent: int = 0):
    """
    Recursively print a concept tree node.

    node: dict returned by build_concept_tree(...)
    token_ds: list-like where token_ds[i] == [token_string, some_meta]
    indent: current indentation level
    """
    prefix = "    " * indent
    # Header for this factor
    log(f"{prefix}Layer {node['layer']}, Concept {node['concept']}:")
    # Its top activating tokens
    for sample_idx, act in node['top_indices']:
        token, meta = token_ds[sample_idx]
        log(f"{prefix}    Token: {token}\t|| Context: {meta} || Act: {act}")
    log(f".")  # blank line before children

    # Recurse into children
    for child in node.get('children', []):
        print_concept_tree(child, token_ds, indent + 1)


# ------------------------------
# Main Code
# ------------------------------


def add_tree_to_graph(G, node, tree_id, model_layer, level=0, parent_id=None):
    node_id = f"T{tree_id}_ML{model_layer}_LV{level}_C{node['concept']}"

    G.add_node(
        node_id,
        model_layer=model_layer,      # transformer / MLP layer
        level=level,                  # depth in tree
        concept_idx=node['concept'],
        top_indices=json.dumps(node.get('top_indices', []))
    )

    if parent_id is not None:
        G.add_edge(parent_id, node_id)

    for child in node.get('children', []):
        add_tree_to_graph(
            G,
            child,
            tree_id=tree_id,
            model_layer=model_layer,
            level=level + 1,
            parent_id=node_id
        )

def analayze_concept_trees(layers, concept_tree_base_path, save_path):
    """
    i.  calculate avergage depth of concept trees per layer
    ii. calculate average branching factor per model layer and per level of tree.
    iii. find depth distribution of concept trees per layer
    """


    for layer in layers:
        stats = {}
        
        # Load the concept tree graph
        graph_file = os.path.join(concept_tree_base_path, f"concept_trees_layer{layer}.graphml")
        if not os.path.isfile(graph_file):
            print(f"Warning: GraphML file not found for layer {layer}: {graph_file}")
            continue
        G = nx.read_graphml(graph_file)

        # Find roots (nodes with in-degree 0)
        roots = [n for n, deg in G.in_degree() if deg == 0]


        all_depths = []
        branching_per_level = defaultdict(list)  # level -> list of branching factors
        leaf_counts = []
        total_nodes_per_tree = []

        for root in roots:
            # BFS traversal to capture levels
            queue = [(root, 0)]
            tree_nodes = 0
            tree_leaves = 0
            max_depth_tree = 0  # track max depth for this tree

            while queue:
                node_id, level = queue.pop(0)
                tree_nodes += 1

                children = list(G.successors(node_id))
                n_children = len(children)

                # track branching factor per level
                branching_per_level[level].append(n_children)

                if n_children == 0:
                    tree_leaves += 1

                # update max depth of this tree
                if level > max_depth_tree:
                    max_depth_tree = level

                for child in children:
                    queue.append((child, level + 1))

            leaf_counts.append(tree_leaves)
            total_nodes_per_tree.append(tree_nodes)
            all_depths.append(max_depth_tree)
        # --- Compute stats ---
        stats['avg_depth'] = np.mean(all_depths) if all_depths else 0
        stats['max_depth'] = np.max(all_depths) if all_depths else 0
        stats['avg_branching_per_level'] = {
                                         int(lvl): float(np.mean(vals)) for lvl, vals in branching_per_level.items()
                                         }

        stats['branching_distribution_per_level'] = {
                                                int(lvl): {int(k): int(v) for k, v in dict(Counter(vals)).items()}
                                                for lvl, vals in branching_per_level.items()
                                                }

        stats['avg_depth'] = float(np.mean(all_depths)) if all_depths else 0
        stats['max_depth'] = int(np.max(all_depths)) if all_depths else 0
        stats['avg_leaf_count_per_tree'] = float(np.mean(leaf_counts)) if leaf_counts else 0
        stats['leaf_ratio_per_tree'] = [
                                    float(leaves) / total if total > 0 else 0
                                    for leaves, total in zip(leaf_counts, total_nodes_per_tree)
                                     ]
        stats['tree_count'] = int(len(roots))
        # Save all of layer's stats to JSON
        layer_save_path = os.path.join(save_path, f"concept_trees_stats_layer{layer}.json")
        os.makedirs(os.path.dirname(layer_save_path), exist_ok=True)

        with open(layer_save_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Analysis saved to {layer_save_path}")




def main():
    """
    1. Train hier SNMF for specified model and layers
    2. Create concept trees per layer
    3. Save concept trees to output path per layer
    4. Analayse concept trees and save analysis to output path per layer
    """
    setup_logging()


    parser = argparse.ArgumentParser(
        description="Run causal interventions from NMF factors; auto-select Gemma vs Regular vector construction."
    )
    parser.add_argument("--model-name", required=True, type=str,
                        help="HF repo id used by HookedTransformer (e.g., 'meta-llama/Llama-3.1-8B' or 'gemma-2-2b').")
    parser.add_argument("--layers", required=True, type=parse_int_list,
                        help="Layers to use, e.g. '0,6,12,18,25' or '0-4'.")
    parser.add_argument("--ranks", required=True, type=parse_int_list,
                        help="Ranks K to iterate, e.g. '100' or '50,100'.")
    parser.add_argument("--factorization-base-path", required=True, type=str,
                        help="Base directory where factorization models live and outputs are written.")
    parser.add_argument("--output-path", required=True, type=str,
                        help="Base output path for concept trees (layer number will be appended).")
    
    args = parser.parse_args()
    layers = args.layers
    ranks = args.ranks
    ranks_str = "-".join(map(str, ranks))
    factorization_base_path = args.factorization_base_path
    output_path = args.output_path
    
    # experimenting with different values for this hyper-parameter, revealed that 0.9 for gemma-2-2b and 0.1 for gpt2-small (that's where pruning appears to be effective)
    minimal_activation = 0.1 if "gemma" in args.model_name.lower() else 0.1

    # 2. Create concept trees per layer
    # 2.1 load hierarchical models once per layer
    nmf_models = load_nmf_decompositions(
        layers=layers,
        factorization_base_path=factorization_base_path,
        ranks=ranks
    )

    # 2.2 create concept trees per layer
    for layer in layers:
        nmf_list = nmf_models[layer]["pretrained_layers"]
        G_trees = nx.DiGraph()
        concept_trees = []
        tree_id = 0
        for i in range(ranks[-1]):  # for each concept in top layer
            concept_tree = build_concept_tree(
                            levels=nmf_list,
                            concept_idx=i,
                            level_idx=len(nmf_list) - 1,
                            top_k_factors=5,
                            top_k_tokens=10,
                            minimal_activation=minimal_activation 
                        )
            concept_trees.append(concept_tree)

            # 3.1 represent each tree as a graph using networkx
            add_tree_to_graph(
            G_trees,
            concept_tree,
            tree_id=tree_id,
            model_layer=layer
            )

            tree_id += 1
        

        # print trees for sanity check
        #for idx, tree in enumerate(concept_trees):
        #    log(f"Concept Tree for Model's Layer {layer}, tree {idx}:")
        #    print_concept_tree(tree, token_ds=nmf_models[layer]["token_dataset"])

        # 3.2 Save concept trees to output path per layer
        os.makedirs(output_path, exist_ok=True)  # create the base folder
        nx.write_graphml(G_trees, os.path.join(output_path, f"concept_trees_layer{layer}.graphml"))

    
    
    # 4. Analayse concept trees and save analysis to output path per layer + saving analysi


    save_path = os.path.join(output_path, "concept_trees_analysis")
    analayze_concept_trees(layers, output_path, save_path)

    log(f"Concept trees analysis is done")




if __name__ == "__main__":
    main()
