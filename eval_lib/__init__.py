"""
eval_lib: A lightweight evaluation library built on top of the Braintrust Eval SDK.

Exports the primary run_eval() entry point and scorer classes for use in eval pipelines.
"""

from eval_lib.runner import run_eval
from eval_lib.types import EvalInput, EvalOutput

__all__ = ["run_eval", "EvalInput", "EvalOutput"]