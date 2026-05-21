"""
Shared type definitions for eval_lib.

This module defines the two core data shapes that flow through the library:

  EvalInput  → a single test case from the dataset, describing what to ask
               and what a correct answer looks like.
  EvalOutput → what the RAG pipeline (or any task function) returns for a
               given EvalInput.

These dataclasses act as the contract between the runner (runner.py), the
task functions supplied by callers, and the scorer implementations in
eval_lib/scorers/. Keeping them in one place makes it easy to extend the
schema without touching scorer or runner logic.
"""

from dataclasses import dataclass, field


@dataclass
class EvalInput:
    """A single test case in the evaluation dataset.

    EvalInput is the "question side" of an eval row. It carries everything
    the task function needs to produce an answer, plus the ground-truth
    expectations that scorers will compare against.

    Attributes:
        query: The natural-language question or query to send to the RAG
            pipeline. For example: "What were Apple's total revenues in Q3 2023?"

        expected: The reference answer a correct pipeline should produce.
            Scorers such as AccuracyScorer and CompletenessScorer compare the
            pipeline's output against this string.

        expected_citations: The citation sources (e.g. document IDs, URLs, or
            formatted labels like "[Doc-3]") that a correct answer must include.
            CitationScorer uses this list to verify that the pipeline actually
            cited the right sources.
    """

    query: str
    expected: str
    expected_citations: list[str] = field(default_factory=list) #citation if its an empty list


@dataclass
class EvalOutput:
    """What the RAG pipeline returns for a given EvalInput.

    EvalOutput is the "answer side" of an eval row. It is produced by the
    task function that the caller passes to run_eval(), and it is handed
    directly to each scorer alongside the corresponding EvalInput.

    Attributes:
        output: The generated answer text returned by the pipeline. This is
            the primary string that AccuracyScorer and CompletenessScorer
            evaluate.

        citations: The citation sources the pipeline included in its response
            (e.g. "[Doc-1]", "https://example.com/source"). CitationScorer
            compares this list against EvalInput.expected_citations to compute
            a citation-coverage score.
    """

    output: str
    citations: list[str] = field(default_factory=list) 