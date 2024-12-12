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
Run `python3 analyze.py` to produce some statistics.
Use `python3 app.py` to produce some more.

### App.py

You'll need to put a `repositories.json` file in the same directory and link to the JSON files produced by `import-prs.py`. Here's an example

```json
{
    "repositories": {
        "Front end": "front-end/pull_requests.json",
        "Back end": "back-end/pull_requests.json"
    }
}
```
