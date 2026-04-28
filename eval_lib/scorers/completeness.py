"""
LLM-as-judge completeness scorer.

Uses a Claude model (via the Anthropic SDK / autoevals) to assess whether the
model output covers all key points required by the expected answer.
Returns a score in [0.0, 1.0] derived from the judge's rating.
"""

from eval_lib.scorers.base import BaseScorer


class CompletenessScorer(BaseScorer):
    pass
