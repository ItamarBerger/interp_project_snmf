import json
import logging
import os
import tempfile
import asyncio
from typing import List, Dict, Set, Optional
from google import genai
from google.genai.types import JobState, UploadFileConfig
import numpy as np
from google.genai.errors import ClientError
from enum import StrEnum

from tracker.initialize_tracker import DEFAULT_ENTITY

logger = logging.getLogger(__name__)

MAX_ENQUEUED_TOKENS = 10000000 # Based on https://ai.google.dev/gemini-api/docs/rate-limits
AGGREGATE_CHAR_COUNT = 3500000 # The number of characters for which we stop and count all the tokens
ACTIVE_BATCH_STATES: Set[str] = {JobState.JOB_STATE_PENDING, JobState.JOB_STATE_RUNNING, JobState.JOB_STATE_QUEUED}
DEFAULT_WAIT_TIME_FOR_NEW_JOB_SUBMISSION = 30* 60 # 30 minutes
JOB_BACKUP_FILE_TEMPLATE = "{batch_name}_parsed_results.json"
DEFAULT_JOB_BACKUP_FOLDER = os.path.join("experiments", "artifacts", "batch_job_backups")
# consider only the last NUM_JOBS_FOR_WAIT_TIME for wait time calculation
NUM_JOBS_FOR_WAIT_TIME = 4
BASE_BACKOFF_FACTOR = 2

class WaitTimeCalcStrategy(StrEnum):
    MEAN = "mean"
    MIN = "min"
    MAX = "max"

class GeminiBatchClient:
    def __init__(self, api_key: str, model_name: str, job_backup_folder: Optional[str] = None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_to_tokens: Dict[str, int] = {}
        self.job_backup_folder = job_backup_folder or DEFAULT_JOB_BACKUP_FOLDER


    # Static methods

    @staticmethod
    def load_and_update_submitted_jobs_file(jobs_backup_path: str, new_batch_name: str) -> bool:
        """
        Loads the submitted jobs file (a file containing a list of batch names in JSON format),
        and appends the new batch name to it.
        """
        try:
            job_history = []
            if os.path.exists(jobs_backup_path):
                with open(jobs_backup_path, 'r') as f:
                    job_history = json.load(f)
            job_history.append(new_batch_name)
            with open(jobs_backup_path, 'w') as f:
                json.dump(job_history, f, indent=2)
            logger.info("Successfully updated file %s with new job %s", jobs_backup_path, new_batch_name)
            return True
        except Exception as e:
            logger.error(f"Failed to update jobs backup file: {e}")
            logger.warning("Your history might be in danger, manually save this job name: %s", new_batch_name)
            return False

    # Internal Methods

    def _get_backup_batch_path(self, batch_name: str) -> str:
        """
        Returns the file path for storing backup parsed results of a batch job.
        We use this in case the job is deleted from the system after some time (google gives us 48 hours to download results).
        """
        if not os.path.exists(self.job_backup_folder):
            os.makedirs(self.job_backup_folder, exist_ok=True)

        safe_batch_name = batch_name.replace("/", "_")
        file_name =  JOB_BACKUP_FILE_TEMPLATE.format(batch_name=safe_batch_name)
        return os.path.join(self.job_backup_folder, file_name)


    def _count_batch_tokens(self, jsonl_filepath):
        """
        Counts total tokens for a JSONL batch file by grouping prompts
        into chunks to stay under the enqueued tokens limit.
        The constant in this file was taken from the official documentation
        """
        total_batch_tokens = 0
        current_chunk = []
        current_chunk_char_count = 0

        logger.info("Counting tokens for batch file %s...", jsonl_filepath)
        with open(jsonl_filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompt_text = data['request']['contents'][0]['parts'][0]['text']

                current_chunk.append(prompt_text)
                current_chunk_char_count += len(prompt_text)

                # 2. If chunk is getting close to 1M tokens (~3.5M characters), count it
                if current_chunk_char_count > AGGREGATE_CHAR_COUNT:
                    total_batch_tokens += self._get_tokens_for_list(current_chunk)
                    current_chunk = []
                    current_chunk_char_count = 0

        # Count the final remaining chunk - or all of them, if we're bellow the AGGREGATE_CHAR_COUNT limit
        if current_chunk:
            total_batch_tokens += self._get_tokens_for_list(current_chunk)

        return total_batch_tokens

    def _get_tokens_for_list(self, text_list):
        """Helper to call the API for a list of strings."""
        try:
            logger.info("Counting tokens for a chunk of %d prompts...", len(text_list))
            response = self.client.models.count_tokens(
                model=self.model_name,
                contents=text_list
            )
            return response.total_tokens
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0

    def _can_submit_new_job(self, new_job_tokens: int) -> bool:
        """
        Checks if a new job with the given token count can be submitted
        without exceeding the MAX_ENQUEUED_TOKENS limit.
        This is only based of what this client instance has in memory,
        so if you restart the program, it will forget previous jobs. That's ok because we also handle rate limits on submission.
        """
        # Get current active batches
        batches = self.client.batches.list()  # Refresh internal state
        active_batches = [batch for batch in batches if batch.state in ACTIVE_BATCH_STATES]

        # Filter the batch_to_tokens to only include active batches
        active_batch_to_tokens = {batch.name: self.batch_to_tokens.get(batch.name, 0) for batch in active_batches}

        current_total = sum(active_batch_to_tokens.values())
        projected_total = current_total + new_job_tokens
        return projected_total <= MAX_ENQUEUED_TOKENS

    def _calculate_wait_time_for_new_submission(self, strategy: WaitTimeCalcStrategy = WaitTimeCalcStrategy.MEAN) -> float:
        """
        Estimates how long we should wait before submitting a new job
        In the case where we exceeded our active enqueued tokens  limit
        This takes into account all successful jobs in the account history.
        Args:
            strategy: The strategy to use for calculating the wait time.
                      Options are 'mean', 'min', 'max'.
        """
        batches = self.client.batches.list()
        # Don't take all - let's just consider the most recent ones
        batches = sorted(batches, key=lambda b: b.create_time, reverse=True)[:NUM_JOBS_FOR_WAIT_TIME]
        successful_batches = [batch for batch in batches if batch.state == JobState.JOB_STATE_SUCCEEDED]
        if not successful_batches:
            return DEFAULT_WAIT_TIME_FOR_NEW_JOB_SUBMISSION

        durations = []
        for batch in successful_batches:
            batch_duration = (batch.end_time - batch.create_time).total_seconds()
            durations.append(batch_duration)


        np_func = getattr(np, strategy)
        estimated_wait_time = np_func(durations)
        logger.info("Calculated estimated wait time for new job submission using strategy '%s': %.2f seconds", strategy, estimated_wait_time)
        return estimated_wait_time




    # External Methods

    async def download_and_save_successful_jobs(self, job_names: list[str], override: bool = False) -> bool:
        """
        Downloads and saves the results of all successful jobs in job_names to local JSON files.
        This is used to back up results in case the whole process does not finish before the jobs are scheduled to be deleted.
        It is recommended to use this method periodically while waiting for jobs to finish.
        Args:
            job_names: A list of job names to check and download if successful.
            override: If True, it will override existing backup files. If False, it will skip downloading if a backup file already exists for a job.
        """
        batches = self.client.batches.list()
        errors = 0
        for batch in batches:
            if batch.state == JobState.JOB_STATE_SUCCEEDED and batch.name in job_names:
                try:
                    backup_filename = self._get_backup_batch_path(batch.name)
                    if os.path.exists(backup_filename) and not override:
                        logger.info(f"Backup file already exists for {batch.name} at {backup_filename}. Skipping download.")
                        continue

                    logger.info(f"Couldn't find backup for {batch.name}. Trying to download results...")
                    parsed_results = await self.retrieve_batch_results(batch.name)
                    with open(backup_filename, 'w') as f:
                        json.dump(parsed_results, f, indent=2)
                    logger.info(f"Saved results for {batch.name} to {backup_filename}")
                except Exception as e:
                    logger.error(f"Failed to retrieve/save results for {batch.name}: {e}")
                    errors += 1
        if errors > 0:
            logger.warning("Couldn't backup all requested batched. Check the logs.")
            return False
        return True

    async def submit_batch_job(self, batch_name: str, prompts_map: Dict[str, str], generation_config: dict, jobs_backup_path: Optional[str] = None) -> str:
        """
        Creates a JSONL file from the prompts, uploads it, and submits a Batch Job.
        Returns the job name (e.g., "projects/.../locations/.../batchJobs/...").
        Args:
            batch_name: A name for the batch job.
            prompts_map: A dict mapping custom IDs to prompt strings.
            generation_config: A dict with generation configuration parameters.
            jobs_backup_path: Optional path to a JSON file where submitted job names will be appended (used for starting the process from a certain point).
        """
        request_ids = list(prompts_map.keys())
        logger.info(f"Preparing batch job for {len(request_ids)} prompts...")

        # Create JSONL content
        batch_file_lines = []
        for rid in request_ids:
            prompt_text = prompts_map[rid]
            entry = {
                "custom_id": str(rid),
                "method": "generateContent",
                "request": {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": generation_config
                }
            }
            batch_file_lines.append(json.dumps(entry))

        # Write to temp file and Upload
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as tmp:
                tmp.write("\n".join(batch_file_lines))
                tmp_path = tmp.name


            # Count the number of tokens in the batch file before uploading
            # For controlling the limits
            total_tokens = self._count_batch_tokens(tmp_path)

            # Check if we can submit the new job, if not, we'll wait and retry
            # This might take a lot of time to complete
            while not self._can_submit_new_job(total_tokens):
                wait_time = self._calculate_wait_time_for_new_submission()
                logger.warning(
                    f"Cannot submit new job '{batch_name}' now. "
                    f"Enqueued tokens limit exceeded. "
                    f"Waiting for {int(wait_time/60)} minutes before retrying..."
                )
                await asyncio.sleep(wait_time)

            logger.info("  Uploading batch file...")
            batch_input_file = self.client.files.upload(
                file=tmp_path,
                config=UploadFileConfig(display_name=batch_name, mime_type='jsonl'))

            # Wait for file to be active
            while batch_input_file.state != "ACTIVE":
                await asyncio.sleep(2)
                batch_input_file = self.client.files.get(name=batch_input_file.name)
                if batch_input_file.state == "FAILED":
                    raise RuntimeError(f"File upload failed: {batch_input_file.state}")

            # Create Batch Job
            logger.info(f"  Submitting Batch Job for file {batch_input_file.name}...")
            created_batch = False
            num_quota_errors = 0
            while not created_batch:
                try:
                    batch_job = self.client.batches.create(
                        model=self.model_name,
                        src=batch_input_file.name,
                    )
                    created_batch = True
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "resource" in err_msg or "exhausted" in err_msg or "quota" in err_msg:
                        num_quota_errors += 1
                        logger.error("Rate limit or quota exceeded while submitting batch job.")
                        # Add wait time - do exponential backoff if we keep hitting quota errors
                        backoff_factor = BASE_BACKOFF_FACTOR ** (num_quota_errors - 1)
                        logger.info("Starting exponential backoff. Number of quota errors so far: %d", num_quota_errors)
                        new_wait_time = self._calculate_wait_time_for_new_submission(WaitTimeCalcStrategy.MIN) * backoff_factor
                        logger.info("Waiting for %d minutes before retrying job submission...", int(new_wait_time/60))
                        await asyncio.sleep(new_wait_time)
                    else:
                        raise e


            # Update backup file if provided
            if jobs_backup_path:
                self.load_and_update_submitted_jobs_file(jobs_backup_path, batch_job.name)

            # Store token count for this batch
            self.batch_to_tokens[batch_job.name] = total_tokens

            logger.info(f"  Job created: {batch_job.name}")
            return batch_job.name

        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}")
            raise e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def wait_for_jobs(self, job_names: List[str], poll_interval: int = 30) -> List[str]:
        """
        Polls the list of job names until ALL are in a terminal state (COMPLETED, FAILED, CANCELLED).
        Returns the list of job names that completed successfully.
        """
        logger.info(f"Starting polling for {len(job_names)} jobs (Interval: {poll_interval}s)...")

        active_jobs = set(job_names)
        completed_successfully = []

        while active_jobs:
            # We iterate over a copy so we can remove from the active set
            current_check_list = list(active_jobs)

            for job_name in current_check_list:
                try:
                    job = self.client.batches.get(name=job_name)

                    # Do not remove only if still active. Partial results are also failures.
                    if not (job.state in ACTIVE_BATCH_STATES):
                        if job.state == JobState.JOB_STATE_SUCCEEDED:
                            logger.info(f"Job {job_name} SUCCEEDED.")
                            completed_successfully.append(job_name)
                        elif job.state in [JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED,
                                           JobState.JOB_STATE_EXPIRED]:
                            logger.error(f"Job {job_name} ended with status: {job.state}")
                        else:
                            logger.warning("Job %s has ended with status %s.", job_name, job.state)
                        active_jobs.remove(job_name)
                except ClientError as e:
                    # We might get a ClientError if the job was deleted from the system because it has been too long
                    # But we might have a backup of the results already saved locally
                    # In this case, we still want to add the job name to the completed_successfully list
                    job_backup_file = self._get_backup_batch_path(job_name)
                    if os.path.exists(job_backup_file):
                        logger.info("Batch %s was deleted but we have backup. Marking as successful.", job_name)
                        completed_successfully.append(job_name)
                        active_jobs.remove(job_name)
                    else:
                        logger.error(f"ClientError while checking status for {job_name}: {e}")
                        logger.warning("No backup file found. Can't mark as successful.")
                except Exception as e:
                    logger.error(f"Error checking status for {job_name}: {e}")

            if active_jobs:
                logger.info(f" Waiting for {len(active_jobs)} jobs to finish...")
                await asyncio.sleep(poll_interval)

        if len(completed_successfully) == len(job_names):
            logger.info("All jobs completed successfully.")
        else:
            logger.warning(f"{len(completed_successfully)}/{len(job_names)} jobs completed successfully.")

        return completed_successfully

    async def retrieve_batch_results(self, job_name: str) -> Dict[str, str]:
        """
        Downloads and parses the results for a completed batch job.
        Returns a dict {custom_id: result_text}.
        """
        results_map = {}
        try:
            # By default - we always try to get "fresh" results.
            # Otherwise we fall back to local backups
            batch_job = self.client.batches.get(name=job_name)
            if batch_job.state != JobState.JOB_STATE_SUCCEEDED:
                logger.warning(f"Cannot retrieve results: Job {job_name} is {batch_job.state}")
                return results_map

            output_file_name = batch_job.dest.file_name
            logger.info(f"  Downloading results from {output_file_name}...")

            # Download content
            file_content = self.client.files.download(file=output_file_name)

            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')

            # Parse JSONL
            for line in file_content.strip().split('\n'):
                if not line: continue
                try:
                    item = json.loads(line)
                    cid = item.get("custom_id")

                    # Extract text from response
                    # Structure: item['response']['candidates'][0]['content']['parts'][0]['text']
                    if 'response' in item and 'candidates' in item['response']:
                        candidates = item['response']['candidates']
                        if candidates:
                            # Safely access parts
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            if parts:
                                text = parts[0].get('text', "")
                                results_map[cid] = text
                except Exception as e:
                    logger.warning(f"Failed to parse result line in {job_name}: {e}")
        except ClientError as e:
                    # See if we already have a backup of the results - and use them.
                    job_backup_file = self._get_backup_batch_path(job_name)
                    if os.path.exists(job_backup_file):
                        logger.info("The batch %s was probably deleted, but the good news is we have backup. Restored from backup file", job_name)
                        with open(job_backup_file, 'r') as f:
                            results_map = json.load(f)
                            return results_map
                    else:
                        logger.error(f"ClientError while parsing results for {job_name}: {e}")
                        logger.warning("No backup file found.Can't return results.")
        except Exception as e:
            logger.error(f"Error retrieving/parsing results for {job_name}: {e}")

        return results_map