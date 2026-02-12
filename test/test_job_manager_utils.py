import json
import os
import pytest
from unittest.mock import patch


class TestEnsureAllPromptsHaveResults:
    """Tests for the ensure_all_prompts_have_results function."""

    @pytest.fixture
    def temp_logs_folder(self, tmp_path):
        """Create a temporary logs folder for testing.

        Uses pathlib for cross-platform compatibility (works on Windows, macOS, Linux).
        The tmp_path fixture is provided by pytest and creates a unique temporary
        directory for each test invocation. The directory is automatically cleaned
        up after the test session.

        Location of temp files:
        - macOS/Linux: /tmp/pytest-of-<user>/pytest-<session>/test_<name><num>/test_logs/
        - Windows: C:\\Users\\<user>\\AppData\\Local\\Temp\\pytest-of-<user>\\pytest-<session>\\test_<name><num>\\test_logs\\
        """
        logs_folder = tmp_path / "test_logs"
        logs_folder.mkdir(parents=True, exist_ok=True)
        return str(logs_folder)

    def test_missing_prompts_saved_to_file(self, temp_logs_folder):
        """Test that missing prompts are detected and saved to a file."""
        # Mock the LOGS_FOLDER constant before importing the function
        with patch("experiments.utils.job_manager_utils.LOGS_FOLDER", temp_logs_folder):
            from experiments.utils.job_manager_utils import ensure_all_prompts_have_results

            # Setup: prompts_map has more entries than results_map
            prompts_map = {
                "prompt_1": "What is the capital of France?",
                "prompt_2": "Explain quantum computing.",
                "prompt_3": "List 3 programming languages.",
                "prompt_4": "What is machine learning?",
            }

            results_map = {
                "prompt_1": "Paris is the capital of France.",
                "prompt_3": "Python, JavaScript, Rust.",
            }

            # Execute
            result = ensure_all_prompts_have_results(prompts_map, results_map)

            # Assert
            assert result is False, "Should return False when prompts are missing"

            # Check that a missing prompts file was created
            log_files = os.listdir(temp_logs_folder)
            assert len(log_files) == 1, "Expected exactly one log file"

            missing_prompts_file = os.path.join(temp_logs_folder, log_files[0])
            assert "_missing_prompts.json" in log_files[0], "File should have _missing_prompts.json suffix"

            # Verify the contents of the missing prompts file
            with open(missing_prompts_file, "r") as f:
                saved_missing_prompts = json.load(f)

            expected_missing = {
                "prompt_2": "Explain quantum computing.",
                "prompt_4": "What is machine learning?",
            }
            assert saved_missing_prompts == expected_missing, "Missing prompts should match expected"

    def test_all_prompts_have_results(self, temp_logs_folder):
        """Test that True is returned when all prompts have results."""
        with patch("experiments.utils.job_manager_utils.LOGS_FOLDER", temp_logs_folder):
            from experiments.utils.job_manager_utils import ensure_all_prompts_have_results

            prompts_map = {
                "prompt_1": "What is AI?",
                "prompt_2": "Define NLP.",
            }

            results_map = {
                "prompt_1": "AI is artificial intelligence.",
                "prompt_2": "NLP is natural language processing.",
            }

            result = ensure_all_prompts_have_results(prompts_map, results_map)

            assert result is True, "Should return True when all prompts have results"

            # No missing prompts file should be created
            log_files = os.listdir(temp_logs_folder)
            assert len(log_files) == 0, "No log file should be created when all prompts have results"

    def test_empty_prompts_map(self, temp_logs_folder):
        """Test with empty prompts_map."""
        with patch("experiments.utils.job_manager_utils.LOGS_FOLDER", temp_logs_folder):
            from experiments.utils.job_manager_utils import ensure_all_prompts_have_results

            prompts_map = {}
            results_map = {"some_result": "value"}

            result = ensure_all_prompts_have_results(prompts_map, results_map)

            assert result is True, "Should return True when prompts_map is empty"

    def test_empty_results_map(self, temp_logs_folder):
        """Test with empty results_map - all prompts should be missing."""
        with patch("experiments.utils.job_manager_utils.LOGS_FOLDER", temp_logs_folder):
            from experiments.utils.job_manager_utils import ensure_all_prompts_have_results

            prompts_map = {
                "prompt_1": "First prompt",
                "prompt_2": "Second prompt",
            }
            results_map = {}

            result = ensure_all_prompts_have_results(prompts_map, results_map)

            assert result is False, "Should return False when results_map is empty"

            # Verify all prompts are in the missing file
            log_files = os.listdir(temp_logs_folder)
            assert len(log_files) == 1

            with open(os.path.join(temp_logs_folder, log_files[0]), "r") as f:
                saved_missing_prompts = json.load(f)

            assert saved_missing_prompts == prompts_map

