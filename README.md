# FinQA Agentic System

An agentic system for the FinQA dataset that generates DSL (Domain Specific Language) programs for financial question answering.

## Overview

The orchestrator LLM receives the **full document context** (pre_text, table, post_text) along with the question, then uses tools to:
1. Retrieve relevant evidence (optional - for filtering)
2. Generate a DSL program using the fine-tuned generator
3. Execute the DSL to verify correctness (optional)

## Requirements

- Python 3.12.3
- Ollama with a model installed (e.g., `qwen2.5:3b`)
- CUDA-compatible GPU (tested on 3090)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langchain-ollama langgraph sentence-transformers transformers torch rich sympy
```

## Models

The system uses three models:

1. **Orchestrator**: Ollama model (e.g., `qwen2.5:3b`) - controls the agent flow
2. **Retriever**: Trained CrossEncoder at `reranker-sweep/jjyfsr7j/checkpoint-1200`
3. **Generator**: Trained Qwen2-5-Coder-3B at `/home/pg2860/hpml/__output__/qwen2-5-coder-3b-instruct-full-input_gold-output_program/final_model`

## Usage

```bash
# Run on 10 samples from dev set
.venv/bin/python -m src.run_finqa --model qwen2.5:3b --split dev --max-samples 10

# Run single query with verbose output
.venv/bin/python -m src.run_finqa --model qwen2.5:3b --single --query-id 0 --verbose

# Run full dev set
.venv/bin/python -m src.run_finqa --model qwen2.5:3b --split dev
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `qwen3:4b` | Ollama model name |
| `--split` | `dev` | Dataset split (train/dev/test) |
| `--max-samples` | 0 | Max samples to run (0=all) |
| `--single` | - | Run single query mode |
| `--query-id` | `0` | Query ID for single mode |
| `--verbose` | - | Show detailed output |
| `--max-tool-calls` | 10 | Max tool calls per query |

## Output

Results are saved to `__output__/exp_<timestamp>/`:
- `config.json` - Experiment configuration
- `results.json` - Full results
- `query_<id>/trace.html` - Visual execution trace
- `query_<id>/messages.json` - Full message history

## Evaluation

Two metrics are reported (using symbolic equivalence via sympy):

1. **Generator Accuracy**: Whether the `generate_dsl` tool's raw output matches the gold program
2. **Agent Accuracy**: Whether the agent's final answer matches the gold program

## Project Structure

```
src/
├── orchestrator.py    # LangGraph agent with tool orchestration
├── finqa_tools.py     # CrossEncoder retriever + Qwen2 generator
├── dsl_executor.py    # DSL parsing, execution, and evaluation
├── finqa_data.py      # Data loading utilities
├── run_finqa.py       # Main runner script
├── logger.py          # Logging infrastructure
└── html_trace_generator.py  # HTML trace visualization
```
