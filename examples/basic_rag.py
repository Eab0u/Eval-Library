"""
Example: evaluating a basic RAG pipeline with eval_lib.

Demonstrates how to define a dataset, a task function that calls a RAG system,
and how to wire up AccuracyScorer, CompletenessScorer, and CitationScorer
before passing everything to run_eval().
"""

import asyncio

from eval_lib import EvalInput, EvalOutput, run_eval
from eval_lib.scorers import AccuracyScorer, CitationScorer, CompletenessScorer


async def mock_rag_pipeline(row: EvalInput) -> EvalOutput:
    # Replace this with your real RAG pipeline call.
    # For demonstration purposes this returns a hard-coded answer.
    return EvalOutput(
        output=f"The answer to '{row.query}' is {row.expected}.",
        citations=row.expected_citations,
    )


dataset = [
    EvalInput(
        query="What is the capital of France?",
        expected="Paris",
        expected_citations=["[Doc-1]"],
    ),
    EvalInput(
        query="What is the capital of Germany?",
        expected="Berlin",
        expected_citations=["[Doc-2]"],
    ),
    EvalInput(
        query="What is the capital of Japan?",
        expected="Tokyo",
        expected_citations=["[Doc-3]"],
    ),
]


async def main():
    results = await run_eval(
        name="basic-rag-example",
        dataset=dataset,
        task=mock_rag_pipeline,
        scorers=[
            AccuracyScorer(),
            CitationScorer(),
            CompletenessScorer(api_key="your-api-key-here"),
        ],
    )

    for result in results:
        print(f"Q: {result.input.query}")
        print(f"A: {result.output.output}")
        print(f"Scores: {result.scores}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
