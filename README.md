# GitHub Pull Request Analysis

I wanted some statistics about pull requests, so I asked various friends of mine (mainly Claude, though his mate Copilot helped out too) to generate some code.

## How to use

Set up your environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Use the `python3 import-prs.py` to grab data.
Run `python3 analyze-prs.py` to produce some statistics.
