import torch
import logging
import wandb
import os

logger = logging.getLogger(__name__)

def save_model_and_artifact(model, cfg, file_path, layer, rank):
    # Save locally
    torch.save(model.state_dict(), file_path)
    logger.info(f"Model state saved to {file_path}")

    # Prepare artifact for wandb
    # Sanitize model name by replacing "/" with "-" (WandB doesn't allow "/" in artifact names)
    sanitized_model_name = cfg['model_name'].replace("/", "-")
    art_name = f"{sanitized_model_name}-{layer}-{rank}-{wandb.run.id}"
    artifact_metadata = {
            "configs": cfg
        }
    artifact = wandb.Artifact(
        name=art_name,
        type="model",
        description=f"Fitted SNMF {model.__class__.__name__} in layer {layer} and rank {rank} with seed {cfg.get('seed')}",
        metadata=artifact_metadata
    )

    # Add all the files under the folder of this file to the artifact
    base_folder = file_path.parent
    for root, _, files in os.walk(base_folder):
        for f in files:
            full_path = os.path.join(root, f)
            artifact.add_file(full_path, f)
    artifact.add_file(file_path)

    # tags
    tags = [cfg["model_name"], f"seed-{cfg['seed']}"]

    # Save the artifact to wandb
    wandb.log_artifact(artifact, tags=tags)