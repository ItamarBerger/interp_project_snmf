import asyncio
import json
import os
import sys
import time
from typing import List
import logging
from .gemini_client import GeminiBatchClient
from .batching import chunk_dict
from constants import LOGS_FOLDER, LogColor


logger = logging.getLogger(__name__)



def load_existing_jobs(submitted_jobs_file: str | None) -> list:
    existing_jobs = []
    if submitted_jobs_file and os.path.exists(submitted_jobs_file):
        try:
            with open(submitted_jobs_file, "r") as f:
                existing_jobs = json.load(f)
            if not isinstance(existing_jobs, list):
                logger.warning("State file content is not a list. Ignoring.")
                existing_jobs = []
            else:
                logger.info(f"Loaded {len(existing_jobs)} existing jobs from {submitted_jobs_file}")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}. Starting fresh.")
            existing_jobs = []
    return existing_jobs


async def submit_batches(prompts_map: dict, client: GeminiBatchClient, existing_jobs: list, submitted_jobs_file: str, batch_size: int, batch_name_prefix: str, generation_config: dict) -> List[str]:
    active_jobs = existing_jobs.copy()
    logger.info(f"===== Phase 1: Submitting {len(prompts_map)} prompts in batches of {batch_size} =====")
    prompt_batches = list(chunk_dict(prompts_map, batch_size))

    # Skip batches that were already submitted
    start_index = len(active_jobs)
    if start_index > 0:
        logger.info(f"  Skipping first {start_index} batches found in {submitted_jobs_file}")
        if start_index >= len(prompt_batches):
            logger.info("All batches appear to have been submitted already.")
            return active_jobs

    for i in range(start_index, len(prompt_batches)):
        batch_prompts = prompt_batches[i]

        try:
            logger.info(f"  Submitting Batch {i + 1}/{len(prompt_batches)}...")
            job_name = await client.submit_batch_job(
                batch_name=f"{batch_name_prefix}_{int(time.time())}_{i}",
                prompts_map=batch_prompts,
                generation_config=generation_config,
            )
            active_jobs.append(job_name)

            # Prevent hitting CreateBatch rate limits
            if i < len(prompt_batches) - 1:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"  Error submitting batch {i + 1}: {e}")
            sys.exit(1)
    return active_jobs


def ensure_all_prompts_have_results(prompts_map: dict, results_map: dict) -> bool:
    missing_prompts = []
    for prompt_id in prompts_map.keys():
        if prompt_id not in results_map:
            missing_prompts.append(prompt_id)

    missing_prompts_file_location = os.path.join(LOGS_FOLDER, f"{int(time.time())}_missing_prompts.json")
    if missing_prompts:
        # Create log folder if it doesn't exist
        os.makedirs(LOGS_FOLDER, exist_ok=True)
        with open(missing_prompts_file_location, "w") as f:
            json.dump(missing_prompts, f, indent=2)
        num_missing_prompts = len(missing_prompts)
        logger.warning(f"{LogColor.RED}->>>> There are {num_missing_prompts} prompts without results.\n Missing prompt IDs have been saved to {missing_prompts_file_location}{LogColor.RESET}")
        return False
    else:
        logger.info(f"{LogColor.GREEN} All prompts have results!{LogColor.RESET}")
        return True

