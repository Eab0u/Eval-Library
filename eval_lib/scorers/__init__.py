"""
Scorer registry for eval_lib.

Re-exports all built-in scorers so callers can import directly from eval_lib.scorers.
"""

from eval_lib.scorers.accuracy import AccuracyScorer
from eval_lib.scorers.citation import CitationScorer
from eval_lib.scorers.completeness import CompletenessScorer, QuestionCompletenessScorer

__all__ = [
    "AccuracyScorer",
    "CitationScorer",
    "CompletenessScorer",
    "QuestionCompletenessScorer",
]
