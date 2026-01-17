from typing import Dict, Any, List
import logging
import json
import sys

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads JSON data from the specified file path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} entries from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from file: {file_path}")
        sys.exit(1)

def calculate_entry_means(entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the mean concept, fluency, and final scores for a single entry.
    Returns a dictionary with the calculated means.
    """
    results = entry.get('sentence_results', [])
    if not results:
        logger.warning(f"No sentence_results found for entry: {entry.get('description', 'Unknown')}")
        return {
            "mean_concept_score": 0.0,
            "mean_fluency_score": 0.0,
            "mean_final_score": 0.0
        }

    total_concept = sum(r.get('concept_score', 0) for r in results)
    total_fluency = sum(r.get('fluency_score', 0) for r in results)
    total_final = sum(r.get('final_score', 0) for r in results)
    count = len(results)

    return {
        "mean_concept_score": total_concept / count,
        "mean_fluency_score": total_fluency / count,
        "mean_final_score": total_final / count
    }
