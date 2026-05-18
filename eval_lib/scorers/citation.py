import re

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
Loops through all of the expected citations, sums together all of the 
ones that are found, and then divides it by the length of the expected citations
to get the ratio of found to total citations. Similar to accuracy, but you 
can get partial credit for some citations. If there is no expected citations
list given, then just return a full score of 1.0
"""

class CitationScorer(BaseScorer):
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        if not input.expected_citations:
            return 1.0
        found = sum(
            1 for citation in input.expected_citations
            if re.search(citation, output.output, re.IGNORECASE)
        )
        return found / len(input.expected_citations)
