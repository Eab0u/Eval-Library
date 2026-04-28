"""
Core eval runner for eval_lib.

This module is the single entry point for running evaluations. Its public
surface is one async function, run_eval(), which orchestrates the full
evaluation lifecycle:

  1. Reads the Braintrust API key from the environment.
  2. Converts the caller's eval_lib types (EvalInput / EvalOutput) into the
     dict-based format that the Braintrust Eval SDK expects.
  3. Wraps each BaseScorer instance into a lightweight callable that matches
     the Braintrust scorer interface.
  4. Delegates execution to braintrust.Eval(), which handles parallelism,
     logging, and result storage in the Braintrust platform.
  5. Converts the raw Braintrust results back into eval_lib's EvalResult
     objects and returns them to the caller.

Separation of concerns
-----------------------
- runner.py owns the Braintrust integration and type-bridging logic.
- Scorer logic lives exclusively in eval_lib/scorers/.
- Data shapes are defined in eval_lib/types.py.

Keeping the Braintrust coupling contained here means that if the upstream
SDK changes its interface, only this file needs to be updated.

Environment variables
----------------------
BRAINTRUST_API_KEY (required): Your Braintrust project API key. Obtain it
    from the Braintrust dashboard. run_eval() raises EnvironmentError at call
    time if this variable is absent or empty, so misconfiguration surfaces
    immediately rather than mid-experiment.
"""

import os
from collections.abc import Awaitable, Callable

from braintrust import Eval

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput, EvalResult

# The environment variable name is defined once here so every reference stays
# in sync and callers can import this constant for documentation purposes.
API_KEY_ENV_VAR = "BRAINTRUST_API_KEY"


def _to_braintrust_data(dataset: list[EvalInput]) -> list[dict]:
    """Convert our EvalInput list to the dict format Braintrust expects.

    Braintrust data rows must be dicts with at least an ``input`` key.
    We also pass ``expected`` at the top level so Braintrust can display
    it in the experiment UI and pass it through to scorers unchanged.
    The full EvalInput object is stored under ``input`` so the task
    function and scorers can access expected_citations and any future fields
    without an extra lookup.
    """
    return [
        {"input": row, "expected": row.expected}
        for row in dataset
    ]


def _wrap_task(
    task: Callable[[EvalInput], Awaitable[EvalOutput]],
) -> Callable[[EvalInput], Awaitable[EvalOutput]]:
    """Return an async callable compatible with braintrust.Eval's task parameter.

    Braintrust passes the ``input`` value from each data row directly to the
    task function. Because we store the full EvalInput object there, no
    additional conversion is needed — the wrapper is a thin identity shim that
    exists solely to document the boundary.
    """
    async def _task_wrapper(eval_input: EvalInput) -> EvalOutput:
        return await task(eval_input)

    return _task_wrapper


def _wrap_scorer(scorer: BaseScorer) -> Callable:
    """Adapt a BaseScorer instance into a Braintrust-compatible scorer callable.

    Braintrust scorers are callables that receive ``output``, ``expected``,
    and ``input`` as keyword arguments and return either a float or a dict
    with ``name`` and ``score`` keys. This wrapper translates between that
    convention and eval_lib's BaseScorer.score() interface.

    The wrapper's ``__name__`` is set to scorer.name so Braintrust uses the
    human-readable scorer name in the experiment UI and result rows.
    """
    def _scorer_wrapper(
        output: EvalOutput,
        expected: str,
        input: EvalInput | None = None,
        **_kwargs,
    ) -> dict:
        raw_score: float = scorer.score(output=output, input=input)
        return {"name": scorer.name, "score": raw_score}

    _scorer_wrapper.__name__ = scorer.name
    return _scorer_wrapper


def _collect_results(
    bt_result_with_summary,
    scorers: list[BaseScorer],
) -> list[EvalResult]:
    """Convert Braintrust's EvalResultWithSummary into a list of EvalResult.

    Each item in bt_result_with_summary.results has:
      .input   → the EvalInput we passed in (stored verbatim by Braintrust)
      .output  → the EvalOutput returned by the task function
      .scores  → a dict mapping scorer name → Score(name, score, ...)

    We extract the numeric .score from each Score object (falling back to a
    direct float cast for forward-compatibility) and build our EvalResult.
    """
    results: list[EvalResult] = []
    for row in bt_result_with_summary.results:
        scores: dict[str, float] = {}
        for scorer_name, score_obj in (row.scores or {}).items():
            if hasattr(score_obj, "score"):
                scores[scorer_name] = float(score_obj.score)
            else:
                scores[scorer_name] = float(score_obj)

        results.append(
            EvalResult(
                input=row.input,
                output=row.output,
                scores=scores,
            )
        )
    return results


async def run_eval(
    name: str,
    dataset: list[EvalInput],
    task: Callable[[EvalInput], Awaitable[EvalOutput]],
    scorers: list[BaseScorer],
) -> list[EvalResult]:
    """Run an evaluation experiment and return scored results.

    This is the primary entry point for eval_lib. It orchestrates the full
    eval lifecycle: type conversion, Braintrust experiment execution, and
    result aggregation.

    The function is async because the Braintrust Eval SDK is async. Call it
    with ``await`` inside an existing event loop, or use ``asyncio.run()``
    at the top level of a script.

    Parameters
    ----------
    name:
        Human-readable experiment name shown in the Braintrust UI. Each call
        with the same name creates a new experiment run under that name, so
        you can track regressions over time.

    dataset:
        The list of test cases to evaluate. Each EvalInput carries the query
        (``input``), the reference answer (``expected``), and the expected
        citation sources (``expected_citations``). The list may be as small as
        a single item for smoke tests or as large as your full benchmark.

    task:
        An async callable that represents the system under test — typically
        your RAG pipeline. It receives one EvalInput and must return an
        EvalOutput. run_eval() calls it once per row in ``dataset``.

        Example signature::

            async def my_rag_pipeline(row: EvalInput) -> EvalOutput:
                answer, citations = await query_rag(row.input)
                return EvalOutput(output=answer, citations=citations)

    scorers:
        One or more BaseScorer instances to apply to every (input, output)
        pair. Each scorer contributes one named float score in [0.0, 1.0] to
        the result. The built-in scorers are AccuracyScorer, CompletenessScorer,
        and CitationScorer; you can also supply custom subclasses.

    Returns
    -------
    list[EvalResult]
        One EvalResult per row in ``dataset``, preserving the original order.
        Each EvalResult contains the original EvalInput, the EvalOutput
        produced by the task, and a ``scores`` dict mapping each scorer's name
        to its numeric score.

    Raises
    ------
    EnvironmentError
        If the ``BRAINTRUST_API_KEY`` environment variable is not set.

    Example
    -------
    ::

        import asyncio
        from eval_lib.runner import run_eval
        from eval_lib.types import EvalInput, EvalOutput
        from eval_lib.scorers import AccuracyScorer, CompletenessScorer

        dataset = [
            EvalInput(
                input="What is the capital of France?",
                expected="Paris",
            ),
        ]

        async def task(row: EvalInput) -> EvalOutput:
            return EvalOutput(output=await my_pipeline(row.input))

        results = asyncio.run(
            run_eval(
                name="geography-v1",
                dataset=dataset,
                task=task,
                scorers=[AccuracyScorer(), CompletenessScorer()],
            )
        )

        for r in results:
            print(r.scores)
    """
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise EnvironmentError(
            f"The {API_KEY_ENV_VAR!r} environment variable is not set. "
            "Obtain your API key from the Braintrust dashboard and export it "
            "before calling run_eval()."
        )

    bt_results = await Eval(
        name,
        data=_to_braintrust_data(dataset),
        task=_wrap_task(task),
        scores=[_wrap_scorer(s) for s in scorers],
        api_key=api_key,
    )

    return _collect_results(bt_results, scorers)
