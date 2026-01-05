import sys, os, argparse, random, numpy as np, torch, pickle
from datetime import datetime
from pathlib import Path
from typing import List

from utils import setup_logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.json_handler import JsonHandler
from llm_utils.activation_generator import ActivationGenerator, extract_token_ids_sample_ids_and_labels
from data_utils.concept_dataset import SupervisedConceptDataset
from factorization.seminmf import NMFSemiNMF  # typing only
import logging

logger = logging.getLogger(__name__)

# ----------------------------- utils -----------------------------
def log(txt: str) -> None:
    logger.info(txt)

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def parse_int_list(spec: str) -> List[int]:
    """
    Accepts: '0,1,2' or '0-31' (inclusive) or '0:32' (end exclusive) or '0:32:2' or mixes like '0,4,10-12'
    """
    spec = spec.strip()
    out: List[int] = []
    for chunk in spec.split(','):
        c = chunk.strip()
        if not c:
            continue
        if ':' in c:  # start:end[:step], end exclusive
            parts = [int(x) for x in c.split(':')]
            if len(parts) == 2: start, end = parts; step = 1
            elif len(parts) == 3: start, end, step = parts
            else: raise argparse.ArgumentTypeError("Range must be start:end or start:end:step")
            out.extend(range(start, end, step))
        elif '-' in c:  # start-end inclusive
            a, b = [int(x) for x in c.split('-', 1)]
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
        end   = min(len(tokens), i + context_window + 1)
        ctx_tokens = [
            act_generator.model.to_str_tokens([tokens[j]])[0][0]
            for j in range(start, end) if sample_ids[j] == sid
        ]
        token_ds.append((token_str, "".join(ctx_tokens)))
    return token_ds

def get_top_activating_indices(G_np: np.ndarray, concept_idx: int, num_samples: int = 10):
    activations, non_zero_indices = [], []
    col = G_np[:, concept_idx]
    top_idx = np.argsort(col)[-num_samples:]
    for i in top_idx:
        a = float(col[i])
        if a <= 0: continue
        activations.append(a)
        non_zero_indices.append(int(i))
    return non_zero_indices, activations

# ----------------------------- main -----------------------------
def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Extract top contexts for Semi-NMF factors from saved models (fully arg-driven).")
    # Required explicit paths (no assumptions)
    p.add_argument("--models-dir", type=str, required=True,
                   help="Directory containing trained models, organized as {models-dir}/{layer}/{rank}/nmf-l{layer}-r{rank}.pkl")
    p.add_argument("--output-json", type=str, required=True,
                   help="Path to the JSON output file to write.")

    # Data / model generation
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--factor-mode", type=str, choices=["mlp","attn","resid"], default="mlp")
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

    args = p.parse_args()
    set_seed(args.seed)

    models_dir = Path(args.models_dir).resolve()
    save_path  = Path(args.output_json).resolve()
    data_path  = Path(args.data_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log("Job started.")
    log(f"Models dir: {models_dir}")
    log(f"Output JSON: {save_path}")
    log(f"Data path: {data_path}")

    # Build generator + dataset
    log(f"Init ActivationGenerator {args.model_name} [{args.factor_mode}] on {args.model_device}")
    act_generator = ActivationGenerator(
        args.model_name,
        model_device=args.model_device,
        data_device=args.data_device,
        mode=args.factor_mode
    )
    dataset = SupervisedConceptDataset(str(data_path))
    tokens, sample_ids, labels = extract_token_ids_sample_ids_and_labels(dataset, act_generator)
    token_context = generate_token_contexts(tokens, sample_ids, act_generator, args.context_window)

    json_handler = JsonHandler(
        ["K", "level", "layer", "h_row", "top_activations", "sparsity"],
        str(save_path),
        auto_write=False
    )
    ranks = args.ranks
    ranks_str = "-".join(map(str, ranks))
    layers = args.layers


    # Load per (layer): {models-dir}/{layer}/hier_snmf-l{layer}-r{rank0}-{}-...-{rankL-1}.pkl
    # Essientially loading hierarchical models once per layer
    nmf_models = {}
    for layer in layers:
        fp = os.path.join(
            models_dir,
            str(layer),
            f"hier_snmf-l{layer}-r{ranks_str}.pkl"
      )
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
        
        pretrained_layers = hier_snmf["pretrained_layers"]  # list of NMFSemiNMF
        for level, nmf in enumerate(pretrained_layers):
            G_np = nmf.G_.detach().cpu().numpy() if isinstance(nmf.G_, torch.Tensor) else nmf.G_
            rank = nmf.H.shape[0]
            for concept_idx in range(rank):
                top_idx, top_acts = get_top_activating_indices(
                    G_np, concept_idx=concept_idx, num_samples=args.num_samples_per_factor
                )
                formatted = [
                    {"token": token_context[i][0], "activation": a, "context": token_context[i][1]}
                    for i, a in zip(top_idx, top_acts)
                ]
                json_handler.add_row(
                    K=rank, level=level, layer=layer, h_row=concept_idx,
                    top_activations=formatted, sparsity=args.sparsity
                )

    json_handler.write()
    log("Done.")

if __name__ == "__main__":
    main()
