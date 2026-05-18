import re

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput


class AccuracyScorer(BaseScorer):
    def __init__(self, pattern: str | None = None):
        self._pattern = pattern

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        pattern = self._pattern if self._pattern is not None else input.expected
        match = re.search(pattern, output.output, re.IGNORECASE)
        return 1.0 if match else 0.0
