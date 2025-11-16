# Simple Evals - Comprehensive Benchmark Suite

A lightweight library for evaluating language models across multiple benchmarks including MMLU, MATH, GPQA, MGSM, DROP, HumanEval, SimpleQA, BrowseComp, and HealthBench.


## Overview

This repository contains implementations of various evaluation benchmarks for language models, with support for both OpenAI, Claude, and local Ollama models.

## Available Benchmarks

### Standard Benchmarks (No Grader Required)

- **MMLU** - Measuring Massive Multitask Language Understanding
- **MATH** - Mathematical Problem Solving (uses equality checker)
- **GPQA** - Graduate-Level Google-Proof Q&A Benchmark
- **MGSM** - Multilingual Grade School Math Benchmark
- **DROP** - Reading Comprehension with Discrete Reasoning
- **HumanEval** - Python Programming Evaluation

### LLM-Graded Benchmarks (Require --grader-model)

- **SimpleQA** - Short-form factuality evaluation
- **BrowseComp** - Browsing agents benchmark
- **HealthBench** - Healthcare language model evaluation
  - `healthbench` - Main dataset
  - `healthbench_hard` - Hard variant
  - `healthbench_consensus` - Consensus variant
- **HealthBenchMeta** - Meta-evaluation for healthcare models

## Setup

### Install Dependencies

```bash
uv pip install -e .
```

### Install Just Command Runner

The `just` command runner is used to download benchmark datasets. Installation instructions: https://github.com/casey/just?tab=readme-ov-file#installation

Quick install:
```bash
# macOS
brew install just

# Linux
cargo install just

# Or download binary from releases
```

### Download Benchmark Data

Most benchmarks load data from URLs automatically, but you can optionally download them locally for better performance:

```bash
# Download all datasets
cd scripts
just download-all

# Or download individual datasets
just download-mmlu
just download-math
just download-gpqa
just download-mgsm
just download-drop
just download-simpleqa
just download-browsecomp
just download-healthbench
just download-healthbench-hard
just download-healthbench-consensus

# List all available commands
just list
```

Data will be downloaded to the `data/` directory.

## Usage

### List Available Models

```bash
uv run python -m simple_evals --list-models
```

Supported models:
- OpenAI: GPT-5, GPT-5.1, GPT-4.1, GPT-4o, GPT-4.5-preview, o1, o3, o4-mini, etc.
- Claude: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.1
- Google Gemini: Gemini 2.5 Pro, Gemini 2.5 Flash
- Ollama: llama3.1, llama3.2, qwen, gemma3, etc.


### List Available Evaluations

```bash
uv run python -m simple_evals --list-evals
```

### Running Evaluations

#### Standard Benchmarks (No Grader)

```bash
# MMLU with GPT-5
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --examples=10

# MMLU with GPT-5 (low reasoning effort for faster execution)
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --reasoning-effort low --examples=10

# MMLU with GPT-4o
uv run python -m simple_evals --eval=mmlu --model=gpt-4o --examples=10

# MATH
uv run python -m simple_evals --eval=math --model=gpt-5 --examples=10

# GPQA
uv run python -m simple_evals --eval=gpqa --model=gpt-5-pro --examples=10

# MGSM
uv run python -m simple_evals --eval=mgsm --model=gpt-5.1 --examples=10

# DROP
uv run python -m simple_evals --eval=drop --model=gpt-5-mini --examples=10

# HumanEval
uv run python -m simple_evals --eval=humaneval --model=gpt-5 --examples=10
```

#### Reasoning Effort Control

For reasoning models (GPT-5, GPT-5.1, o3, o4-mini, etc.), you can control the reasoning effort:

```bash
# Fastest execution - Use 'minimal' for GPT-5, 'none' for GPT-5.1
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --reasoning-effort minimal --examples=10
uv run python -m simple_evals --eval=mmlu --model=gpt-5.1 --reasoning-effort none --examples=10

# Low - Fast execution with some reasoning (~3x speedup vs default)
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --reasoning-effort low --examples=10

# Medium - Balanced (default)
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --examples=10

# High - Best quality, slower execution
uv run python -m simple_evals --eval=mmlu --model=gpt-5 --reasoning-effort high --examples=10
```

**Performance Comparison (2 examples):**
- Default (medium): ~40 seconds
- `--reasoning-effort low`: ~13 seconds (3x faster)
- `--reasoning-effort minimal/none`: ~5-8 seconds (fastest, 5-8x speedup)
- Mini/Nano models default to low reasoning effort

**Note:** GPT-5.1 uses `none` for the lowest effort, while GPT-5 uses `minimal`. Both provide the fastest execution.

#### LLM-Graded Benchmarks (Require Grader)

```bash
# SimpleQA
uv run python -m simple_evals --eval=simpleqa --model=gpt-4o --grader-model=gpt-4o --examples=10

# BrowseComp
uv run python -m simple_evals --eval=browsecomp --model=gpt-4o --grader-model=gpt-4o --examples=10

# HealthBench
uv run python -m simple_evals --eval=healthbench --model=gpt-4o --grader-model=gpt-4o --examples=10
uv run python -m simple_evals --eval=healthbench_hard --model=gpt-4o --grader-model=gpt-4o --examples=10
uv run python -m simple_evals --eval=healthbench_consensus --model=gpt-4o --grader-model=gpt-4o --examples=10
```

#### With Claude Models

```bash
# Claude Haiku 4.5 (fastest and cheapest)
uv run python -m simple_evals --eval=mmlu --model=claude-haiku-4-5 --examples=10

# Claude Sonnet 4.5 (best for complex tasks)
uv run python -m simple_evals --eval=mmlu --model=claude-sonnet-4-5 --examples=10

# With grader model
uv run python -m simple_evals --eval=healthbench --model=claude-sonnet-4-5 --grader-model=claude-sonnet-4-5 --examples=10
```

#### Claude Extended Thinking

Claude 4.1+ models support extended thinking for enhanced reasoning on complex tasks. Use `--thinking-budget` to control the token budget for internal reasoning:

```bash
# Enable extended thinking with moderate budget (1k-2k tokens for most tasks)
uv run python -m simple_evals --eval=mmlu --model=claude-sonnet-4-5 --thinking-budget 2048 --examples=10

# Higher budget for complex reasoning (math, coding)
uv run python -m simple_evals --eval=math --model=claude-opus-4-1 --thinking-budget 10000 --examples=10

# Fast model with thinking
uv run python -m simple_evals --eval=gpqa --model=claude-haiku-4-5 --thinking-budget 1024 --examples=10

# Maximum reasoning for very complex tasks
uv run python -m simple_evals --eval=math --model=claude-sonnet-4-5 --thinking-budget 32000 --examples=10

# Without thinking (default, faster and cheaper)
uv run python -m simple_evals --eval=mmlu --model=claude-sonnet-4-5 --examples=10
```

**Token Budget Guidelines:**
- **Minimum:** 1,024 tokens (required by Claude API)
- **1k-2k:** Simple reasoning tasks, general Q&A
- **2k-8k:** Standard problem solving, comparison tasks
- **8k-16k:** Complex multi-step reasoning, coding tasks
- **16k-32k+:** Advanced mathematics, complex algorithms, deep analysis

**Notes:**
- Claude may use less than the allocated budget
- Thinking tokens are billed as output tokens
- Higher budgets improve quality but increase latency and cost
- Default behavior (no `--thinking-budget`): thinking is disabled
- When thinking is enabled, temperature is automatically set to 1.0
- max_tokens is automatically adjusted to be greater than thinking budget
- Supported models: Claude Sonnet 4.5, Haiku 4.5, Opus 4.1

**Reference:** [Claude Extended Thinking Documentation](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)

#### With Google Gemini Models

```bash
# Gemini 2.5 Flash (fast and efficient)
uv run python -m simple_evals --eval=mmlu --model=gemini-2.5-flash --examples=10

# Gemini 2.5 Pro (most capable)
uv run python -m simple_evals --eval=mmlu --model=gemini-2.5-pro --examples=10

# With thinking budget (enable reasoning)
# 0 = disable thinking, -1 = dynamic thinking, or specific token count
# Pro: 128-32768, Flash: 0-24576
uv run python -m simple_evals --eval=mmlu --model=gemini-2.5-pro --thinking-budget 1024 --examples=10

# With grader model
uv run python -m simple_evals --eval=healthbench --model=gemini-2.5-pro --grader-model=gemini-2.5-pro --examples=10
```

#### With Ollama Models (Requires Ollama Running)

```bash
# Start Ollama first
ollama serve

# Run evaluation
uv run python -m simple_evals --eval=mmlu --model=llama3.1 --examples=10
uv run python -m simple_evals --eval=healthbench --model=llama3.1 --grader-model=llama3.1 --examples=10
```

### Using OpenAI-Compatible Endpoints

You can run evaluations against any OpenAI-compatible API endpoint using `--openai-base-url`. This allows you to use local models via Ollama or vLLM, or alternative providers like OpenRouter.

#### With Ollama (OpenAI-Compatible Endpoint)

Ollama provides an OpenAI-compatible API that can be used with any OpenAI model configuration:

```bash
# Start Ollama
ollama run qwen3:1.7b

# Use --model-override to specify the exact model name for your endpoint
uv run python -m simple_evals \
  --eval=mmlu \
  --model=gpt-4o \
  --openai-base-url=http://localhost:11434/v1/ \
  --model-override=llama3.2 \
  --examples=10
```

**Reference:** [Ollama OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)

#### With vLLM (OpenAI-Compatible Server)

vLLM provides an OpenAI-compatible server for running local models at high performance:

```bash
# Start vLLM server (example with Llama 3.2)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --port 8000

uv run python -m simple_evals \
  --eval=mmlu \
  --model=gpt-4o \
  --openai-base-url=http://localhost:8000/v1 \
  --model-override=meta-llama/Llama-3.2-3B-Instruct \
  --examples=10
```

**References:**
- [vLLM OpenAI Chat Completion Client](https://docs.vllm.ai/en/stable/examples/online_serving/openai_chat_completion_client.html)
- [vLLM Quickstart: OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server)

#### With OpenRouter

OpenRouter provides access to multiple LLM providers through a unified OpenAI-compatible API:

```bash
uv run python -m simple_evals \
  --eval=mmlu \
  --model=gpt-4o \
  --use-openrouter \
  --examples=10

# Use --model-override to specify OpenRouter-specific models
uv run python -m simple_evals \
  --eval=mmlu \
  --model=gpt-4o \
  --use-openrouter \
  --model-override=deepseek/deepseek-chat \
  --examples=10
```

**Note:** Check [OpenRouter's model list](https://openrouter.ai/models) for available models and their exact names.

#### With Custom Docker/Local Endpoints

For custom OpenAI-compatible endpoints (like Docker containers), use both flags together:

```bash
uv run python -m simple_evals \
  --eval=mmlu \
  --model=gpt-4o \
  --openai-base-url=http://localhost:12434/engines/llama.cpp/v1/ \
  --model-override=ai/gemma3 \
  --examples=10
```

**Key Points:**
- `--openai-base-url`: Sets the API endpoint URL
- `--model-override`: Specifies the exact model name your endpoint expects
- `--model`: Selects which evaluation configuration to use (gpt-4o, gpt-4.1, etc.)

### Multiple Models

```bash
# Evaluate multiple models
uv run python -m simple_evals --eval=mmlu --model=gpt-4o,claude-sonnet-4-5 --examples=10

# Use ensemble grading
uv run python -m simple_evals --eval=healthbench --model=gpt-4o --grader-model=gpt-4o,claude-opus-4-1 --examples=10
```

### Debug Mode

```bash
# Run with minimal examples for debugging
uv run python -m simple_evals --eval=mmlu --model=gpt-4o --debug
```

### Advanced Options

```bash
# Specify number of examples
--examples=100

# Set number of repeats (for certain evals)
--n-repeats=5

# Set number of threads (for HealthBench)
--n-threads=120
```

## Environment Variables

Use `.env` file to set environment variables. Check the `.env.example` file for the required variables.

## Data Files

All benchmarks load data from URLs by default, but can be downloaded locally using the justfile commands for better performance.

## Notes

- Most benchmarks load data from OpenAI's public blob storage
- Local data files in `data/` are used if available, otherwise data is fetched from URLs
- The MATH benchmark uses an equality checker (GPT-4o by default) to verify mathematical answers
- HealthBench and similar benchmarks require a grader model for LLM-based evaluation
- Use `--debug` mode to quickly test with minimal examples

## Attribution

This fork is based on OpenAI's simple-evals: https://github.com/openai/simple-evals

And `kayzi0` simple-evals fork: https://github.com/kayzi0/simple-evals

Individual benchmark citations and licenses can be found in the original repository.

## License

MIT License - See LICENSE file for details
