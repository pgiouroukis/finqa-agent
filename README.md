# FinQA Agentic System

An agentic system for the FinQA dataset that generates DSL (Domain Specific Language) programs for financial question answering.

## Overview

The orchestrator LLM receives the **full document context** (pre_text, table, post_text) along with the question, then uses tools to:
1. Retrieve relevant evidence
2. Generate a DSL program using the fine-tuned generator
3. Execute the DSL to verify correctness

## Requirements

- Python 3.11.8 (see Setup)
- Ollama with a model installed (e.g., `qwen2.5:3b`)
- CUDA-compatible GPU (run on RTX 3090 GPU)

## Setup

We will use `micromamba` (installation [guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)) for package-management. Please also add `conda-forge` for setting-up environments.

    micromamba config append channels conda-forge

Create a new environment. (`[VERSION]` use 3.11.8).

    micromamba create -n [PROJECT_NAME] python=[VERSION]

Activate environment.

    micromamba activate [PROJECT_NAME]

You will need to install these:

    micromamba install -c conda-forge python=[VERSION]

Install requirements.

    pip install -r requirements.txt

Installing Ollama.

    brew install ollama
    ollama serve

Running the script.

    ollama pull [MODEL]
    python -m agent.run_finqa --model [MODEL] --split dev --max-samples [MAX_SAMPLES]

## Models

The system uses three models:

1. **Orchestrator**: Ollama model (e.g., `qwen2.5:3b`) - controls the agent flow
2. **Retriever**: Trained CrossEncoder at [HF](https://huggingface.co/svk2118/reranker-22m)
3. **Generator**: Trained Qwen2-5-Coder-3B at [HF](https://huggingface.co/petrosg/qwen2-5-coder-3b-instruct-full-input_gold-output_program)

## Usage

```bash
# Run on 10 samples from dev set
python -m agent.run_finqa --model [MODEL] --split dev --max-samples 10

# Run single query with verbose output
python -m agent.run_finqa --model [MODEL] --single --query-id 0 --verbose

# Run full dev set
python -m agent.run_finqa --model [MODEL] --split dev
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

## Evaluation

Two metrics are reported (using symbolic equivalence via sympy):

1. **Generator Accuracy**: Whether the `generate_dsl` tool's raw output matches the gold program
2. **Agent Accuracy**: Whether the agent's final answer matches the gold program

## Project Structure

```
.
├── README.md                       (current file)
├── agent                           (main directory containing agent)
│   ├── __init__.py
│   ├── dsl                         (DSL generator)
│   │   ├── executor.py             (DSL execution and validation scripts)
│   │   ├── model                   (model directory for DSL generator)
│   │   │   └── ...
│   │   └── run.py
│   ├── orchestrator.py             (Agent orchestrator - LangChain ReAct)
│   ├── preprocess.py               (FinQA dataset pre-processing scripts)
│   ├── retriever                   (Retriever)
│   │   ├── model                   (model directory for cross-encoder reranker)
│   │   │   └── ...
│   │   └── run.py
│   └── run_finqa.py                (entry point for application)
├── data                            (FinQA dataset)
│   ├── dev.json
│   ├── private_test.json
│   ├── test.json
│   └── train.json
├── experiments                     (Experiments, training, misc. scripts)
│   ├── finetune_gemma.py
│   ├── retriever.ipynb             (Training notebook for retriever)
│   └── tmp
│       ├── agent.py
│       ├── logger.py
│       └── trace_generator.py
└── requirements.txt                (pip requirements)
```
