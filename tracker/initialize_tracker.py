import wandb
from .utils import get_run_name

DEFAULT_PROJECT = "snmf-hierarchical-steering"
DEFAULT_ENTITY = "inter-proj"

def init_tracker(cfg: dict, layer) -> wandb.sdk.wandb_run.Run:
    tags = [cfg["model_name"], f"sparsity-{str(cfg['sparsity'])}", f"ranks-{cfg['ranks']}", f"seed-{cfg['seed']}"]
    return wandb.init(
        project=cfg.get("wandb_project", DEFAULT_PROJECT),
        entity=cfg.get("wandb_entity", DEFAULT_ENTITY),
        name=get_run_name(cfg, layer),
        config=cfg,
        mode=cfg.get("wandb_mode", "online"),
        tags=tags,
        job_type=cfg.get("wandb_job_type", "single-run")
    )