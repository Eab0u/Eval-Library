from abc import ABC, abstractmethod

from eval_lib.types import EvalInput, EvalOutput


class BaseScorer(ABC):
    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        ...
