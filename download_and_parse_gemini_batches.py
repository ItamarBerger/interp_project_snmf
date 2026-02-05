import os
import json
import asyncio
from experiments.utils import GeminiBatchClient
from utils import setup_logging
import argparse
import logging
import sys

logger = logging.getLogger(__name__)

async def download_and_save_jobs(batch_client, job_names):
    result = await batch_client.download_and_save_successful_jobs(job_names)

    if result:
        logger.info("Downloaded and saved all successful jobs.")
    else:
        logger.error("Could not download all successful jobs.")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download and save successful Gemini batch jobs.")
    parser.add_argument("--submitted-jobs-file", type=str, required=True,
                        help="Path to the JSON file containing submitted job names.")
    parser.add_argument("--backup-folder", type=str, required=True)
    args = parser.parse_args()

    submitted_jobs_file = args.submitted_jobs_file

    batch_client = GeminiBatchClient(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-2.0-flash",
                                     submitted_jobs_path=submitted_jobs_file, job_backup_folder=args.backup_folder)

    job_names = []

    if os.path.exists(submitted_jobs_file):
        logger.info("Found submitted jobs file, loading job names...")
        with open(submitted_jobs_file, "r") as f:
            job_names = json.load(f)
    else:
        logger.error(f"Submitted jobs file {submitted_jobs_file} does not exist.")
        sys.exit(1)

    asyncio.run(download_and_save_jobs(batch_client, job_names))


if __name__ == "__main__":
    main()