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

## Braintrust API key

Braintrust is optional. If no API key is provided, evals run locally and
results are returned directly — nothing is logged to Braintrust.

If you do want to log to Braintrust, there are three ways to supply the key:

**Option 1 — Pass it directly to `run_eval()`**:
```python
await run_eval(..., api_key="your-key-here")
```

**Option 2 — Set it for the current terminal session**:
```powershell
# Windows (PowerShell)
$env:BRAINTRUST_API_KEY = "your-key-here"

# macOS / Linux
export BRAINTRUST_API_KEY="your-key-here"
```

**Option 3 — Set it permanently for your user account**:
```powershell
# Windows (PowerShell)
[System.Environment]::SetEnvironmentVariable("BRAINTRUST_API_KEY", "your-key-here", "User")

# macOS / Linux — add this line to ~/.zshrc or ~/.bashrc
export BRAINTRUST_API_KEY="your-key-here"
```

Your API key can be found in the [Braintrust dashboard](https://www.braintrustdata.com).

## Anthropic API key

`CompletenessScorer` and `QuestionCompletenessScorer` call a Claude model to judge
completeness, so they require an Anthropic API key. The same three options apply:

**Option 1 — Pass it directly to the scorer**:
```python
CompletenessScorer(api_key="your-key-here")
QuestionCompletenessScorer(api_key="your-key-here")
```

**Option 2 — Set it for the current terminal session**:
```powershell
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "your-key-here"

# macOS / Linux
export ANTHROPIC_API_KEY="your-key-here"
```

**Option 3 — Set it permanently for your user account**:
```powershell
# Windows (PowerShell)
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your-key-here", "User")

# macOS / Linux — add this line to ~/.zshrc or ~/.bashrc
export ANTHROPIC_API_KEY="your-key-here"
```

Your API key can be found in the [Anthropic console](https://console.anthropic.com).

`AccuracyScorer` and `CitationScorer` use regex only and do not require an Anthropic key.

## Quick start

```python
from eval_lib import run_eval
from eval_lib.scorers import AccuracyScorer, CompletenessScorer, CitationScorer

# coming soon
```
