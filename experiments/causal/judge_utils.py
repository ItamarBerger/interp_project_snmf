import re
import os
import json
import sys
import logging
import time
import asyncio
from typing import List, Generator
from experiments.utils import GeminiBatchClient

logger = logging.getLogger(__name__)

def extract_rating(response_text):
    """
    Extracts the rating from the LLM response.
    Supports both "Rating: [[score]]" and "Rating: score" formats.
    Returns None if extraction fails instead of raising an exception.
    """
    try:
        match = re.search(r'Rating:\s*(?:\[\[(\d+)\]\]|(\d+))', response_text)
        if match:
            score_str = match.group(1) if match.group(1) is not None else match.group(2)
            return int(score_str)
        else:
            logger.info(f"⚠ Warning: Could not extract rating from response: {response_text[:100]}...")
            return None
    except Exception as e:
        logger.info(f"⚠ Warning: Error extracting rating: {e}")
        return None

def chunk_dict(data: dict, size: int) -> Generator[dict, None, None]:
    """Yields chunks of the dictionary with a maximum size."""
    keys = list(data.keys())
    for i in range(0, len(keys), size):
        yield {k: data[k] for k in keys[i:i + size]}


async def submit_batches(prompts_map: dict, client: GeminiBatchClient, existing_jobs: list, args, judge_type: str) -> List[str]:
    active_jobs = existing_jobs.copy()
    logger.info(f"Phase 1: Submitting {len(prompts_map)} prompts in batches...")
    prompt_batches = list(chunk_dict(prompts_map, args.batch_size))
    submitted_jobs_file = args.submitted_jobs_file

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
                batch_name=f"{judge_type}_judge_batch_{int(time.time())}_{i}",
                prompts_map=batch_prompts,
                generation_config={"temperature": 0.0},
                jobs_backup_path=submitted_jobs_file
            )
            active_jobs.append(job_name)

            # Prevent hitting CreateBatch rate limits
            if i < len(prompt_batches) - 1:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"  Error submitting batch {i + 1}: {e}")
            sys.exit(1)
    return active_jobs


def load_existing_jobs(args) -> list:
    existing_jobs = []
    if args.submitted_jobs_file and os.path.exists(args.submitted_jobs_file):
        try:
            with open(args.submitted_jobs_file, "r") as f:
                existing_jobs = json.load(f)
            if not isinstance(existing_jobs, list):
                logger.warning("State file content is not a list. Ignoring.")
                existing_jobs = []
            else:
                logger.info(f"Loaded {len(existing_jobs)} existing jobs from {args.submitted_jobs_file}")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}. Starting fresh.")
            existing_jobs = []
    return existing_jobs