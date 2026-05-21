"""Requires Braintrust auth, converts EvalInput to EvalCase, calls Braintrust Eval, returns Braintrust result."""

import os
from collections.abc import Awaitable, Callable

from braintrust import Eval, EvalCase

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput


def _to_braintrust_data(dataset: list[EvalInput]) -> list[EvalCase]:
    """Convert EvalInput list to Braintrust EvalCase objects."""
    return [EvalCase(input=row, expected=row.expected) for row in dataset]


async def run_eval(
    name: str,
    dataset: list[EvalInput],
    task: Callable[[EvalInput], Awaitable[EvalOutput]],
    scorers: list[BaseScorer],
):
    """Run an evaluation experiment via Braintrust and return Braintrust's result.

    Parameters
    ----------
    name:
        Passed directly to Braintrust Eval() as both the project name and
        experiment name.

    dataset:
        Test cases to evaluate. Each EvalInput carries the query, reference
        answer, and expected citation sources.

    task:
        Async callable representing the system under test. Receives one
        EvalInput and must return an EvalOutput.

    scorers:
        BaseScorer instances to apply to every (input, output) pair.

    Raises
    ------
    EnvironmentError
        If BRAINTRUST_API_KEY is not set in the environment.
    """
    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise EnvironmentError(
            "BRAINTRUST_API_KEY is not set. Export it before running evaluations."
        )

    return await Eval(
        name,
        data=_to_braintrust_data(dataset),
        task=task,
        scores=scorers,
        experiment_name=name,
    )
