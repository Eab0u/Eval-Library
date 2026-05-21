import re

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
Loops through all of the expected citations, sums together all of the
ones that are found, and then divides it by the length of the expected citations
to get the ratio of found to total citations. Similar to accuracy, but you
can get partial credit for some citations.

Raises ValueError if expected_citations is empty.
Returns a float rounded to 2 decimal places.
"""

class CitationScorer(BaseScorer):
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        """Score citation coverage as a ratio of found to expected citations.

        Returns:
            Float in [0.0, 1.0], rounded to 2 decimal places.

        Raises:
            ValueError: If input.expected_citations is empty.
        """
        if not input.expected_citations:
            raise ValueError(
                "expected_citations is empty. Provide at least one expected citation."
            )
        actual_answer: str = output.output
        total_citations: int = len(input.expected_citations)
        found: int = sum(
            1 for citation in input.expected_citations
            if re.search(citation, actual_answer, re.IGNORECASE)
        )
        return round(found / total_citations, 2)
