import torch
import logging
import wandb

logger = logging.getLogger(__name__)

def save_model_and_artifact(model, cfg, file_path, layer, rank):
    # Save locally
    torch.save(model.state_dict(), file_path)
    logger.info(f"Model state saved to {file_path}")

    # Prepare artifact for wandb
    art_name = f"{cfg['model_name']}-{layer}-{rank}-{wandb.run.id}"
    artifact_metadata = {
            "configs": cfg
        }
    artifact = wandb.Artifact(
        name=art_name,
        type="model",
        description=f"Fitted SNMF {model.__class__.__name__} in layer {layer} and rank {rank} with seed {cfg.get('seed')}",
        metadata=artifact_metadata
    )

    # We might want to add more files
    artifact.add_file(file_path)

    # tags
    tags = [cfg["model_name"], f"seed-{cfg['seed']}"]

    # Save the artifact to wandb
    wandb.log_artifact(artifact, tags=tags)