import os
import re

import anthropic

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
The completeness scorer uses an LLM (claude sonnet) to judge how complete the given
answer was. This requires an API key from the user to use this feature, and if it 
is not given, then an environment error will be raised to deal with this. There 
are 2 independent subscorers that can be individually be used by the user. 
CompletenessScorer: Checks how well the answer covers all of the points in the expected answer.
QuestionCompletenessScorer: Checks how well the answer adresses the question in general.
"""

_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
_MODEL = "claude-sonnet-4-5"
_FLOAT_PATTERN = re.compile(r"\b(1\.0|0\.\d+)\b")

def _resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or os.environ.get(_API_KEY_ENV_VAR)
    if not resolved:
        raise EnvironmentError(
            f"The {_API_KEY_ENV_VAR!r} environment variable is not set. "
            "Obtain your API key from the Anthropic dashboard."
        )
    return resolved


"""
Pulls the exact score out of the models response in the case of extra garbage
Example:
"The score is 0.8"
return 0.8;
"""
def _extract_score(text: str) -> float:
    match = _FLOAT_PATTERN.search(text)
    return float(match.group(1)) if match else 0.0

"""
The CompletenessScorer scores how completely the actual output covers
everything in the expected answer. Given the expected response and the actual response, 
the LLM will judge this score using the users given API key.
"""

class CompletenessScorer(BaseScorer):
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key))

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        prompt = (
            f"Expected answer: {input.expected}\n"
            f"Actual output: {output.output}\n\n"
            "Score how completely the actual output covers all key points in the "
            "expected answer. Respond with only a float between 0.0 and 1.0."
        )
        message = self._client.messages.create(
            model=_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_score(message.content[0].text)

"""
The QuestionCompletenessScorer scores how well the answer addresses the question.
Given the question and response, the LLM will judge this score using the users given API key.
"""
class QuestionCompletenessScorer(BaseScorer):
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key))

    def score(self, output: EvalOutput, input: EvalInput) -> float:
        prompt = (
            f"Question: {input.input}\n"
            f"Actual output: {output.output}\n\n"
            "Score how completely the actual output addresses the question. "
            "Respond with only a float between 0.0 and 1.0."
        )
        message = self._client.messages.create(
            model=_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_score(message.content[0].text)
