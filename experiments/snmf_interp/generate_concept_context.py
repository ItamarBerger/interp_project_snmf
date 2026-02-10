import sys, os, argparse, random, numpy as np, torch, pickle
from collections import Counter
from pathlib import Path
import json
from typing import List, Optional
import networkx as nx

from utils import setup_logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.evaluation.json_handler import JsonHandler
from experiments.evaluation.concept_trees.concept_tree_utils import build_concept_tree, get_node_id
from llm_utils.activation_generator import ActivationGenerator, extract_token_ids_sample_ids_and_labels
from data_utils.concept_dataset import SupervisedConceptDataset
from factorization.seminmf import NMFSemiNMF  # typing only
import logging

logger = logging.getLogger(__name__)

TREE_OUTPUT_PATH = os.path.join("{concept_tree_base}", "layer_{layer}", "concept_tree_{tree_id}.graphml")


# ----------------------------- utils -----------------------------
def log(txt: str) -> None:
    logger.info(txt)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(spec: str) -> List[int]:
    """
    Accepts: '0,1,2' or '0-31' (inclusive) or '0:32' (end exclusive) or '0:32:2' or mixes like '0,4,10-12'
    """
    spec = spec.strip()
    out: List[int] = []
    for chunk in spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        if ":" in c:  # start:end[:step], end exclusive
            parts = [int(x) for x in c.split(":")]
            if len(parts) == 2:
                start, end = parts
                step = 1
            elif len(parts) == 3:
                start, end, step = parts
            else:
                raise argparse.ArgumentTypeError("Range must be start:end or start:end:step")
            out.extend(range(start, end, step))
        elif "-" in c:  # start-end inclusive
            a, b = [int(x) for x in c.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(c))
    return out


def generate_token_contexts(tokens, sample_ids, act_generator, context_window: int):
    token_ds = []
    for i in range(len(tokens)):
        sid = sample_ids[i]
        token_str = act_generator.model.to_str_tokens([tokens[i]])[0][0]
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        ctx_tokens = [act_generator.model.to_str_tokens([tokens[j]])[0][0] for j in range(start, end) if sample_ids[j] == sid]
        token_ds.append((token_str, "".join(ctx_tokens)))
    return token_ds



def lists_equal_unordered(l1, l2):
    # For simple dicts with hashable values
    return Counter(map(frozenset, [d.items() for d in l1])) == \
           Counter(map(frozenset, [d.items() for d in l2]))

def add_node_data_to_layer_feature_dict(
    node, layer, feature_dict, token_contexts, concept_idx, sparsity, tree_id: str, pretrained_levels: List[NMFSemiNMF]
):
    # Recursively add node data to JSON handler.
    level = node["level"]
    top_activations = []
    for sample_idx, activation in node.get("top_indices", []):
        token_str, context_str = token_contexts[sample_idx]
        top_activations.append({"token": token_str, "context": context_str, "activation": activation})

    key = (layer, level, concept_idx)
    if not key in feature_dict:
        feature_dict[key] = {
            "K": pretrained_levels[level].H.shape[0],
            "level": level,
            "layer": layer,
            "h_row": concept_idx,
            "tree_ids": [tree_id],
            "top_activations": top_activations,
            "sparsity": sparsity,
        }
    else:
        if not lists_equal_unordered(feature_dict[key]["top_activations"], top_activations):
            logger.info("The activations are different, so we're adding another entry to the feature dict with the same key but different tree_id")
            new_key = (layer, level, concept_idx, tree_id) # It doesn't matter that it's not the same key format, we're going to use only the values
            feature_dict[new_key] = {
                "K": pretrained_levels[level].H.shape[0],
                "level": level,
                "layer": layer,
                "h_row": concept_idx,
                "tree_ids": [tree_id],
                "top_activations": top_activations,
                "sparsity": sparsity,
            }
        else:
            feature_dict[key]["tree_ids"].append(tree_id)

    for child in node.get("children", []):
        add_node_data_to_layer_feature_dict(
            node=child,
            layer=layer,
            feature_dict=feature_dict,
            token_contexts=token_contexts,
            concept_idx=child["concept"],
            sparsity=sparsity,
            tree_id=tree_id,
            pretrained_levels=pretrained_levels,
        )


def build_nx_tree(tree: nx.DiGraph, node: dict, layer: int, level: int, parent_id: Optional[str] = None):
    """
    Recursively build tree as nx.DiGraph.
    """
    concept_idx = node["concept"]
    node_id = get_node_id(tree.graph["tree_id"], layer, level, concept_idx)

    top_indices = node.get("top_indices", [])
    # Add root node
    tree.add_node(
        node_id,
        layer=layer,
        level=level,
        concept_idx=concept_idx,
        top_indices=json.dumps(top_indices),
    )

    if parent_id is not None:
        tree.add_edge(parent_id, node_id)

    for child in node.get("children", []):
        build_nx_tree(tree=tree, node=child, layer=layer, level=child["level"], parent_id=node_id)


def write_graphml_trees(nx_trees: List[nx.DiGraph], layer: int, output_folder: str):
    for tree in nx_trees:
        tree_id = tree.graph["tree_id"]
        output_file = TREE_OUTPUT_PATH.format(concept_tree_base=output_folder, layer=layer, tree_id=tree_id)
        # Create folder structure if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(tree, output_file)
        log(f"Saved concept tree {tree_id} to {output_file}")


# ----------------------------- main -----------------------------
def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Extract top contexts for Semi-NMF factors from saved models (fully arg-driven).")
    # Required explicit paths (no assumptions)
    p.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing trained models, organized as {models-dir}/{layer}/{rank}/nmf-l{layer}-r{rank}.pkl",
    )
    p.add_argument("--output-json", type=str, required=True, help="Path to the JSON output file to write.")

    # Data / model generation
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--factor-mode", type=str, choices=["mlp", "attn", "resid"], default="mlp")
    p.add_argument("--data-path", type=str, default="data/hier_concepts.json")

    # Selection
    p.add_argument("--layers", type=parse_int_list, default=parse_int_list("0:32"))
    p.add_argument("--ranks", type=parse_int_list, default=parse_int_list("100"))

    # Extraction behavior
    p.add_argument("--num-samples-per-factor", type=int, default=25)
    p.add_argument("--context-window", type=int, default=15)
    p.add_argument("--sparsity", type=float, default=0.01)  # bookkeeping only
    p.add_argument("--seed", type=int, default=42)

    # Devices
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--model-device", type=str, default=default_dev)
    p.add_argument("--data-device", type=str, default="cpu")
    p.add_argument(
        "--concept-trees-folder",
        type=str,
        default=None,
        help="Folder to save concept trees for reference. By default this is relative to the output location",
    )
    p.add_argument("--top-k-factors", type=int, default=5, help="How many child factors to keep at each level when building concept trees.")
    p.add_argument(
        "--minimal-activation", type=float, default=0.1, help="Minimal activation threshold for including child factors when building concept trees."
    )

    args = p.parse_args()
    set_seed(args.seed)

    models_dir = Path(args.models_dir).resolve()
    save_path = Path(args.output_json).resolve()
    data_path = Path(args.data_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.concept_trees_folder is None:
        args.concept_trees_folder = os.path.join(Path(args.output_json).parent, "concept_trees")

    log("Job started.")
    log(f"Models dir: {models_dir}")
    log(f"Output JSON: {save_path}")
    log(f"Data path: {data_path}")

    # Build generator + dataset
    log(f"Init ActivationGenerator {args.model_name} [{args.factor_mode}] on {args.model_device}")
    act_generator = ActivationGenerator(args.model_name, model_device=args.model_device, data_device=args.data_device, mode=args.factor_mode)
    dataset = SupervisedConceptDataset(str(data_path))
    tokens, sample_ids, labels = extract_token_ids_sample_ids_and_labels(dataset, act_generator)
    token_context = generate_token_contexts(tokens, sample_ids, act_generator, args.context_window)

    json_handler = JsonHandler(["K", "level", "layer", "h_row", "tree_ids", "top_activations", "sparsity"], str(save_path), auto_write=False)
    ranks = args.ranks
    ranks_str = "-".join(map(str, ranks))
    layers = args.layers

    # Load per (layer): {models-dir}/{layer}/hier_snmf-l{layer}-r{rank0}-{}-...-{rankL-1}.pkl
    # Essientially loading hierarchical models once per layer
    nmf_models = {}
    layer_feature_dict = {}  # (layer, level, concept_idx) -> {K, layer, level, h_row, top_activations, tree_ids, sparsity}
    for layer in layers:
        fp = os.path.join(models_dir, str(layer), f"hier_snmf-l{layer}-r{ranks_str}.pkl")
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                nmf_models[layer] = pickle.load(f)
            log(f"Loaded hierarchical NMF for layer {layer}, ranks {ranks_str}")
        else:
            log(f"Missing hierarchical file for layer {layer}: {fp}")
            continue
    for layer in layers:
        hier_snmf = nmf_models.get(layer)
        if hier_snmf is None:
            continue

        pretrained_levels = hier_snmf["pretrained_layers"]  # list of NMFSemiNMF levels

        # First, build the concept trees for this layer (model layer)
        # We will save the results of the concept trees in a json file, that we can later use for analysis and plotting
        # After building concept trees for each layer (model layer), we will generate the concept_context.file as before,
        # only now the top activations will be relative to the hierarchical decomposition, and the context that we will send

        # start with the highest hierarchical level (lowest rank) and go down to the lowest level (highest rank)
        for level_idx in reversed(range(len(pretrained_levels))):
            rank = pretrained_levels[level_idx].H.shape[0]
            nx_trees = []
            for concept_idx in range(rank):  # concept_idx is the index of the concept in the highest level
                tree = build_concept_tree(
                    levels=pretrained_levels,
                    concept_idx=concept_idx,
                    level_idx=level_idx,  # start at the highest level
                    top_k_factors=args.top_k_factors,  # hyperparam: how many child factors to keep at each level
                    top_k_tokens=args.num_samples_per_factor,  # how many top tokens to keep for each factor
                    minimal_activation=args.minimal_activation,  # threshold for including child factors based on activation
                )
                tree_id = f"root_l{layer}_K{rank}_LV{level_idx}_c{concept_idx}"
                # Build nx tree
                nx_tree = nx.DiGraph(tree_id=tree_id)
                build_nx_tree(
                    tree=nx_tree,
                    node=tree,
                    layer=layer,
                    level=level_idx,
                    parent_id=None,
                )
                nx_trees.append(nx_tree)

                # Add to concept context file
                add_node_data_to_layer_feature_dict(
                    node=tree,
                    layer=layer,
                    feature_dict=layer_feature_dict,
                    token_contexts=token_context,
                    concept_idx=concept_idx,
                    sparsity=args.sparsity,
                    tree_id=tree_id,
                    pretrained_levels=pretrained_levels,
                )
            write_graphml_trees(nx_trees, layer, args.concept_trees_folder)

    for key, val in layer_feature_dict.items():
        json_handler.add_row(**val)
    json_handler.write()
    log("Done.")


if __name__ == "__main__":
    main()
