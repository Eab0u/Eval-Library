"""
Regex-based accuracy scorer.

Checks whether the model output matches an expected pattern or exact string
using regular expressions. Returns 1.0 on a full match, 0.0 otherwise.
"""

from eval_lib.scorers.base import BaseScorer


class AccuracyScorer(BaseScorer):
    pass
