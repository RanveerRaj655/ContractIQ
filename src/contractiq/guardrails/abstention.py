"""
guardrails/abstention.py
------------------------
Decides whether to abstain from LLM generation based on retrieval confidence.
"""

from contractiq.config import settings

def should_abstain(top_scores: list[float], threshold: float = None) -> bool:
    """
    Returns True if the retrieval confidence is too low to attempt generation.
    """
    if not top_scores:
        return True
        
    if threshold is None:
        threshold = settings.abstention_threshold
        
    return max(top_scores) < threshold
