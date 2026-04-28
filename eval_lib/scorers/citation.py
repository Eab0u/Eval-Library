"""
Regex-based citation scorer.

Verifies that the model output contains properly formatted citations matching
an expected pattern (e.g., [1], [Doc-A]). Scores by the fraction of required
citations that are present in the output.
"""

from eval_lib.scorers.base import BaseScorer


class CitationScorer(BaseScorer):
    pass
