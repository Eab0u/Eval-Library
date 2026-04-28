"""
Abstract base class for eval_lib scorers.

Defines the BaseScorer interface that all scorers must implement.
Each scorer receives an EvalOutput and expected value, and returns a numeric score
in [0.0, 1.0] along with optional metadata.
"""


class BaseScorer:
    pass
