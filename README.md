# eval_lib

A lightweight Python evaluation library built on top of the [Braintrust Eval SDK](https://www.braintrustdata.com/docs).

## Structure

```
eval_lib/
  __init__.py         # package exports
  runner.py           # run_eval() entry point
  types.py            # shared dataclasses
  scorers/
    base.py           # BaseScorer interface
    accuracy.py       # regex-based accuracy scorer
    completeness.py   # LLM-as-judge completeness scorer
    citation.py       # regex-based citation scorer
examples/
  basic_rag.py        # example RAG eval pipeline
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from eval_lib import run_eval
from eval_lib.scorers import AccuracyScorer, CompletenessScorer, CitationScorer

# coming soon
```
