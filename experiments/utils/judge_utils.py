import re
import logging

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


def harmonic_mean(scores):
    """
    Computes the harmonic mean of the provided scores.
    If any score is zero or None, the harmonic mean will be zero.
    """
    # Filter out None values and convert to list
    valid_scores = [s for s in scores if s is not None]

    # If any scores were None or if any valid score is 0, return 0
    if len(valid_scores) < len(scores) or any(score == 0 for score in valid_scores):
        return 0

    # If no valid scores, return 0
    if not valid_scores:
        return 0

    return len(valid_scores) / sum(1.0 / score for score in valid_scores)