import os
import re

import litellm

from eval_lib.scorers.base import BaseScorer
from eval_lib.types import EvalInput, EvalOutput

"""
The completeness scorer uses an LLM to judge how complete the given answer was.
The model is read from the EVAL_LIB_MODEL environment variable, so any provider
supported by LiteLLM can be used (e.g. OpenAI, Anthropic, Gemini) by setting
that variable to the appropriate model string. The corresponding API key must
also be set as an environment variable (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY).
There are 2 independent scorers:
CompletenessScorer: Checks how well the answer covers all of the points in the expected answer.
QuestionCompletenessScorer: Checks how well the answer addresses the question in general.
"""

_FLOAT_PATTERN = re.compile(r"\b(1\.0|0\.\d+)\b")


def _get_model() -> str:
    model = os.environ.get("EVAL_LIB_MODEL")
    if not model:
        raise EnvironmentError(
            "EVAL_LIB_MODEL is not set. Set it to any LiteLLM-supported model string "
            "(e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001') and ensure the matching "
            "API key environment variable is also set."
        )
    return model


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
The model is determined by the EVAL_LIB_MODEL environment variable.
"""
class CompletenessScorer(BaseScorer):
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        actual_answer: str = output.output
        messages = [
            {
                "role": "system",
                "content": (
                    "Your goal is to score how completely an actual output covers all key points "
                    "in an expected answer. You will be given an expected answer and an actual output. "
                    "Respond with only a float between 0.0 and 1.0, where 1.0 means the output fully "
                    "covers every key point and 0.0 means it covers none."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<expected>The Eiffel Tower is located in Paris, France. "
                    "It was built in 1889 and stands 330 meters tall.</expected>\n"
                    "<output>The Eiffel Tower is in Paris.</output>"
                ),
            },
            {"role": "assistant", "content": "0.3"},
            {
                "role": "user",
                "content": (
                    f"<expected>{input.expected}</expected>\n"
                    f"<output>{actual_answer}</output>"
                ),
            },
        ]
        response = litellm.completion(model=_get_model(), messages=messages, max_tokens=16)
        llm_response: str = response.choices[0].message.content
        return _extract_score(llm_response)


"""
The QuestionCompletenessScorer scores how well the answer addresses the question.
The LLM judges this score based on the question and actual output only.
The model is determined by the EVAL_LIB_MODEL environment variable.
"""
class QuestionCompletenessScorer(BaseScorer):
    def score(self, output: EvalOutput, input: EvalInput) -> float:
        actual_answer: str = output.output
        messages = [
            {
                "role": "system",
                "content": (
                    "Your goal is to score how completely an actual output addresses a given question. "
                    "You will be given a question and an actual output. "
                    "Respond with only a float between 0.0 and 1.0, where 1.0 means the output fully "
                    "addresses the question and 0.0 means it does not address it at all."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<question>What year was the Eiffel Tower built and how tall is it?</question>\n"
                    "<output>The Eiffel Tower was built in 1889 and stands 330 meters tall.</output>"
                ),
            },
            {"role": "assistant", "content": "1.0"},
            {
                "role": "user",
                "content": (
                    f"<question>{input.query}</question>\n"
                    f"<output>{actual_answer}</output>"
                ),
            },
        ]
        response = litellm.completion(model=_get_model(), messages=messages, max_tokens=16)
        llm_response: str = response.choices[0].message.content
        return _extract_score(llm_response)
