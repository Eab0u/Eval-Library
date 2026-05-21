from abc import ABC, abstractmethod

from eval_lib.types import EvalInput, EvalOutput

"""
Base is an interface / Abstrace Base Class that is meant to act
as a blueprint for the actual scorers. It contains a constructor / property
and an abstract method that scores depending on the scorer, returning a float.
"""

class BaseScorer(ABC):
    """Interface / Abstract Base Class that acts as a blueprint for all scorers.

    Provides a default name property, enforces that subclasses implement
    score(), and exposes __call__ in the format Braintrust expects.
    """

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        ...

    def __call__(
        self,
        output: EvalOutput,
        expected: str,
        input: EvalInput | None = None,
        **_kwargs,
    ) -> dict:
        """Braintrust scorer interface: receives output/expected/input, returns name+score."""
        return {"name": self.name, "score": self.score(output=output, input=input)}
