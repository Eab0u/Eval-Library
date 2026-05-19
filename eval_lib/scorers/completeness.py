import re

import litellm

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
The completeness scorer uses an LLM to judge how complete the given answer was.
The model is configurable — any provider supported by LiteLLM can be used
(e.g. OpenAI, Anthropic, Gemini) by passing a model string.
There are 2 independent scorers:
CompletenessScorer: Checks how well the answer covers all of the points in the expected answer.
QuestionCompletenessScorer: Checks how well the answer addresses the question in general.
"""

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_FLOAT_PATTERN = re.compile(r"\b(1\.0|0\.\d+)\b")


"""
Pulls the exact score out of the model's response in the case of extra text.
Example: "The score is 0.8" → returns 0.8
Returns 0.0 if no valid float is found.
"""
def _extract_score(text: str) -> float:
    match = _FLOAT_PATTERN.search(text)
    return float(match.group(1)) if match else 0.0


"""
The CompletenessScorer scores how completely the actual output covers
everything in the expected answer. The LLM judges this score.
"""
class CompletenessScorer(BaseScorer):
    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._api_key = api_key

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        prompt = (
            f"Expected answer: {input.expected}\n"
            f"Actual output: {output.output}\n\n"
            "Score how completely the actual output covers all key points in the "
            "expected answer. Respond with only a float between 0.0 and 1.0."
        )
        response = litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            api_key=self._api_key,
        )
        return _extract_score(response.choices[0].message.content)


"""
The QuestionCompletenessScorer scores how well the answer addresses the question.
The LLM judges this score based on the question and actual output only.
"""
class QuestionCompletenessScorer(BaseScorer):
    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._api_key = api_key

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        prompt = (
            f"Question: {input.input}\n"
            f"Actual output: {output.output}\n\n"
            "Score how completely the actual output addresses the question. "
            "Respond with only a float between 0.0 and 1.0."
        )
        response = litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            api_key=self._api_key,
        )
        return _extract_score(response.choices[0].message.content)
