import re

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
The accuracy scorer tells us the complete accuracy of the output,
either a 1 for accurate or a 0 for not. It does this using a regex matching
the expected output with the actual output and ignoring match case.

Example:
input.expected: The revenue was 4.2 billion
output.output: 4.2 billion
return 1.0;
"""

class AccuracyScorer(BaseScorer):
    def __init__(self, pattern: str = ""):
        self._pattern = pattern

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        pattern: str = self._pattern if self._pattern else input.expected
        actual_answer: str = output.output
        match = re.search(pattern, actual_answer, re.IGNORECASE)
        return 1.0 if match else 0.0
