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

#### With Google Gemini Models

```bash
# Gemini 2.5 Flash (fast and efficient)
uv run python -m simple_evals --eval=mmlu --model=gemini-2.5-flash --examples=10

# Gemini 2.5 Pro (most capable)
uv run python -m simple_evals --eval=mmlu --model=gemini-2.5-pro --examples=10

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
