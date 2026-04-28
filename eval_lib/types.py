"""
Shared type definitions for eval_lib.

This module defines the three core data shapes that flow through the library:

  EvalInput  → a single test case from the dataset, describing what to ask
               and what a correct answer looks like.
  EvalOutput → what the RAG pipeline (or any task function) returns for a
               given EvalInput.
  EvalResult → the fully scored record produced by the runner after all
               scorers have been applied to an (EvalInput, EvalOutput) pair.

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
        input: The natural-language question or query to send to the RAG
            pipeline. For example: "What were Apple's total revenues in Q3 2023?"

        expected: The reference answer a correct pipeline should produce.
            Scorers such as AccuracyScorer and CompletenessScorer compare the
            pipeline's output against this string.

        expected_citations: The citation sources (e.g. document IDs, URLs, or
            formatted labels like "[Doc-3]") that a correct answer must include.
            CitationScorer uses this list to verify that the pipeline actually
            cited the right sources.
    """

    input: str
    expected: str
    expected_citations: list[str] = field(default_factory=list)


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


@dataclass
class EvalResult:
    """The fully scored record for a single (EvalInput, EvalOutput) pair.

    EvalResult is assembled by the runner after all scorers have been applied.
    It bundles the original input, the pipeline's output, and the numeric
    scores from every scorer into a single object that Braintrust can ingest
    and that callers can inspect programmatically.

    Attributes:
        input: The original EvalInput test case that was evaluated, preserved
            here so results are self-contained and can be analysed without
            rejoining against the source dataset.

        output: The EvalOutput produced by the task function for this test
            case.

        scores: A mapping from scorer name to a float score in [0.0, 1.0].
            Keys are the human-readable names of each scorer (e.g.
            "accuracy", "completeness", "citation"). A score of 1.0 means
            the scorer considered the output fully correct; 0.0 means it
            failed entirely. Partial scores in between are allowed and
            encouraged where the scorer supports them.
    """

    input: EvalInput
    output: EvalOutput
    scores: dict[str, float] = field(default_factory=dict)
