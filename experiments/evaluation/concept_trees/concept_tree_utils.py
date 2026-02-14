import json
from typing import Optional, Any

import networkx as nx
import numpy as np
import logging
import pickle
import os
from numpy import ndarray, dtype


logger = logging.getLogger(__name__)
MIN_TOTAL_MASS = 1e-6  # to avoid division by zero in top-p calculations


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


def get_top_activating_indices_hierarchical(
        levels: list,
        concept_idx: int,
        level_idx: int,
        num_samples: int = 10,
        minimal_activation: float = 0.0
):
    """
    For a hierarchical NMF defined by `levels` (where
      levels[0].W has shape (n_samples,   r1),
      levels[1].W has shape (r1,          r2),
      …,
      levels[L].W has shape (r_L,         hidden_dim)
    ), compute the “effective activation” of each sample for
    the concept `concept_idx` at level `level_idx` by
    multiplying

       M = W₀ @ W₁ @ … @ W_{level_idx}

    then taking column `concept_idx` of M.

    Parameters
    ----------
    levels : list of NMFMU
      Pretrained levels 0…L, each with `.W` as a torch.Tensor.
    concept_idx : int
      Which concept/column to inspect at the specified level.
    level_idx : int
      Index of the level (0-based).
    num_samples : int
      How many top‐activating samples to return.
    minimal_activation : float
      Threshold: ignore any sample with activation ≤ this.

    Returns
    -------
    top_indices : list[int]
      The sample indices (0…n_samples-1) of the top activations.
    activations : list[float]
      The corresponding activation values.
    """
    # 1) Multiply W0 @ W1 @ … @ W_level_idx
    M = levels[0].W  # (n_samples, r1)
    for i in range(1, level_idx + 1):
        M = M @ levels[i].W

    # 2) Extract the desired concept column
    #    M has shape (n_samples, r_{level_idx+1})
    col = M[:, concept_idx]  # (n_samples,)
    vals = col.detach().cpu().numpy()  # numpy array for sorting

    # 3) Get indices of the top activations
    sorted_idxs = np.argsort(vals)[::-1]  # descending order
    top_indices = []
    activations = []

    for idx in sorted_idxs:
        if vals[idx] <= minimal_activation:
            break
        top_indices.append(int(idx))
        activations.append(float(vals[idx]))
        if len(top_indices) >= num_samples:
            break

    return top_indices, activations

def build_concept_tree(
    levels: list,
    concept_idx: int,
    level_idx: int,
    top_k_factors: int = 3,
    top_k_tokens: int = 10,
    minimal_activation: float = 0.0
):
    # 1) Get the raw top activations (no filtering)
    indices, acts = get_top_activating_indices_hierarchical(
        levels, concept_idx, level_idx,
        num_samples=top_k_tokens * 5,  # grab extra so filtering has room
        minimal_activation=0.0
    )

    # 2) Compute a per-node threshold
    if 0.0 < minimal_activation < 1.0:
        node_max = max(acts) if acts else 0.0
        token_thresh = minimal_activation * node_max
    else:
        token_thresh = minimal_activation

    # 3) Filter token activations and cap at top_k_tokens
    filtered_tokens = [(i, a) for i, a in zip(indices, acts) if a >= token_thresh]
    filtered_tokens.sort(key=lambda x: x[1], reverse=True)
    filtered_tokens = filtered_tokens[:top_k_tokens]

    node = {
        'level': level_idx,
        'concept': concept_idx,
        'top_indices': filtered_tokens,
        'children': []
    }

    # 4) Recurse for factor-level children if not at bottom
    if level_idx > 0:
        W = levels[level_idx].W.detach().cpu().numpy()
        contrib = W[:, concept_idx]  # each lower-level factor’s importance

        # compute child threshold
        if 0.0 < minimal_activation < 1.0:
            child_thresh = minimal_activation * contrib.max()
        else:
            child_thresh = minimal_activation

        # collect all valid factors, then cap at top_k_factors
        sorted_factors = np.argsort(contrib)[::-1]
        valid_factors = [f for f in sorted_factors if contrib[f] >= child_thresh]
        top_factors = valid_factors[:top_k_factors]

        # stop this branch if there's only one child to create
        if len(top_factors) <= 1:
            return node

        for f_idx in top_factors:
            child = build_concept_tree(
                levels,
                concept_idx=int(f_idx),
                level_idx=level_idx - 1,
                top_k_factors=top_k_factors,
                top_k_tokens=top_k_tokens,
                minimal_activation=minimal_activation
            )
            node['children'].append(child)

    return node


def get_top_p_factors_indices(contributions: np.ndarray, top_p: float) -> ndarray[Any, dtype[Any]]:
    """Get the concept indices of the top-p contributing factors."""
    total_mass = contributions.sum()
    if total_mass < MIN_TOTAL_MASS:
        return np.array([])  # avoid division by zero, treat as no contributors

    probs = contributions / total_mass

    sorted_idxs = np.argsort(probs)[::-1] # From highest to lowest
    sorted_probs = probs[sorted_idxs]
    # Get the cumulative sum of the sorted probabilities [p1, p1+p2, p1+p2+p3, ...] when p1 is the highest
    cumulative_probs = np.cumsum(sorted_probs)
    # Search for the cutoff index where cumulative probability exceeds top_p, starting from the left (highest contributors)
    cutoff_idx = np.searchsorted(cumulative_probs, top_p)
    # Make sure the cutoff doesn't go beyond the array length
    cutoff_idx = min(cutoff_idx, probs.size - 1)

    # Return the indices of the top contributors that together account for at least top_p of the mass
    return sorted_idxs[:cutoff_idx + 1]



def build_concept_tree_top_p(
    levels: list,
    concept_idx: int,
    level_idx: int,
    top_k_tokens: int = 10,
    top_p: float = 0.9,
    minimal_activation: float = 0.0
):
    # 1) Get the raw top activations (no filtering)
    indices, acts = get_top_activating_indices_hierarchical(
        levels, concept_idx, level_idx,
        num_samples=top_k_tokens * 5,  # grab extra so filtering has room
        minimal_activation=0.0
    )

    # 2) Compute a per-node threshold
    if 0.0 < minimal_activation < 1.0:
        node_max = max(acts) if acts else 0.0
        token_thresh = minimal_activation * node_max
    else:
        token_thresh = minimal_activation

    # 3) Filter token activations and cap at top_k_tokens
    filtered_tokens = [(i, a) for i, a in zip(indices, acts) if a >= token_thresh]
    filtered_tokens.sort(key=lambda x: x[1], reverse=True)
    filtered_tokens = filtered_tokens[:top_k_tokens]

    node = {
        'level': level_idx,
        'concept': concept_idx,
        'top_indices': filtered_tokens,
        'children': []
    }

    # 4) Recurse for factor-level children if not at bottom
    if level_idx > 0:
        W = levels[level_idx].W.detach().cpu().numpy()
        contrib = W[:, concept_idx]  # each lower-level factor’s importance

        top_factors = get_top_p_factors_indices(contrib, top_p)

        # stop this branch if there's only one child to create
        if len(top_factors) <= 1:
            return node

        for f_idx in top_factors:
            child = build_concept_tree(
                levels,
                concept_idx=int(f_idx),
                level_idx=level_idx - 1,
                top_k_tokens=top_k_tokens,
                minimal_activation=minimal_activation
            )
            node['children'].append(child)

    return node


def load_nmf_decompositions(layers: list[int], factorization_base_path: str, ranks: list[int]):
    # 2.1 load hierarchical models once per layer
    nmf_models = {}
    ranks_str = "-".join(map(str, ranks))
    for layer in layers:
        fp = os.path.join(
            factorization_base_path,
            str(layer),
            f"hier_snmf-l{layer}-r{ranks_str}.pkl"
      )
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                nmf_models[layer] = pickle.load(f)
            logger.info(f"Loaded hierarchical NMF for layer {layer}, ranks {ranks_str}")
        else:
            logger.warning(f"Missing hierarchical file for layer {layer}: {fp}")

    return nmf_models


def get_node_id(tree_id: int, layer: int, level: int, concept_idx: int):
    return f"Tree[{tree_id}]_L{layer}_LV{level}_C{concept_idx}"


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
