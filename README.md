## Simple Evals - HealthBench

To run a sample evaluation:

1. Make sure you have the required data files downloaded
2. Run with your preferred model and grader

### Setup

The HealthBench data file has been downloaded to `Data/2025-05-07-06-14-12_oss_eval.jsonl`.

### Examples

**With OpenAI models:**
```bash
uv run python -m simple_evals --eval=healthbench --model=gpt-4o --grader-model=gpt-4o --examples=1
```

**With Ollama (requires Ollama to be running):**
```bash
uv run python -m simple_evals --eval=healthbench --model=llama3.1 --grader-model=llama3.1 --examples=1
```

**List available models:**
```bash
uv run python -m simple_evals --list-models
```

**List available evaluations:**
```bash
uv run python -m simple_evals --list-evals
```
