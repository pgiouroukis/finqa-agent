# HPML Project: Small Agent Pipelines for Multi-Step Numeric Financial Tasks

## Team Information

* **Members**:
  * Petros Stylianos Giouroukis (pg2860)
  * Shiv Kampani (svk2118)
  * Kendall Ma (wm2544)
  * Karina Nirmal (kkn2118)

---

## 1. Problem Statement

We study whether small, open-source language models can solve multi-step numerical questions over long financial reports without relying on expensive, large-scale models. Using the FINQA benchmark, we build a modular pipeline consisting of (i) a lightweight cross-encoder retriever to select relevant sentences and table rows, (ii) a fine-tuned generator that outputs either a numeric answer or an executable DSL program, and (iii) an execution-based verifier that validates and runs the program to obtain the final result.

Across three small LLMs and multiple input and supervision settings, we find that **program generation decisively outperforms direct numerical prediction**, **code-specialized models transfer best to DSL synthesis**, and **retrieval quality strongly affects downstream accuracy**. Wrapping these components in a ReAct-style agent further improves end-to-end performance, with a larger controller enabling better tool orchestration and self-correction. All training and inference were engineered to run on a single RTX 3090 (24GB), demonstrating a practical path toward on-premise, low-memory financial QA systems.

An agentic system for the FinQA dataset that generates DSL (Domain Specific Language) programs for financial question answering.

The orchestrator LLM receives the **full document context** (pre_text, table, post_text) along with the question, then uses tools to:

1. Retrieve relevant evidence
2. Generate a DSL program using the fine-tuned generator
3. Execute the DSL to verify correctness

---

## 2. Model Description

### System Overview

The system consists of three learned components and an agent controller:

1. **Orchestrator**

   * Ollama-hosted LLM (e.g., `qwen2.5:3b`, `qwen2.5:14b`)
   * Controls tool selection, iteration, and self-correction (ReAct-style)

2. **Retriever**

   * Lightweight transformer **cross-encoder reranker** (~22.7M params)
   * Scores query–sentence / query–row pairs jointly
   * HuggingFace model:
     [https://huggingface.co/svk2118/reranker-22m](https://huggingface.co/svk2118/reranker-22m)

3. **Generator**

   * Fine-tuned small LLM that outputs either:

     * A final numeric answer, or
     * An executable **FinQA DSL program**
   * Best-performing model:
     [https://huggingface.co/petrosg/qwen2-5-coder-3b-instruct-full-input_gold-output_program](https://huggingface.co/petrosg/qwen2-5-coder-3b-instruct-full-input_gold-output_program)

4. **Executor / Verifier**

   * Executes generated DSL programs
   * Detects malformed programs and arithmetic errors
   * Enables execution-guided correction by the agent

---

## 3. Final Results Summary

All results below are taken directly from experiments reported in the paper.

### 3.1 Retriever Performance

**Metric:** nDCG@10 against gold evidence annotations
**Latency:** measured per query on a single consumer GPU

| Method                                     |    nDCG@10 | Mean Lat (ms) |  P50 (ms) |  P95 (ms) |
| ------------------------------------------ | ---------: | ------------: | --------: | --------: |
| BM25                                       |     0.7182 |          0.76 |      0.73 |      1.17 |
| BM25 (Sectionwise)                         |     0.8438 |          0.73 |      0.72 |      1.04 |
| Pretrained Cross-Encoder                   |     0.7929 |         14.92 |     13.31 |     24.78 |
| Fine-Tuned Cross-Encoder                   |     0.9148 |         14.16 |     12.92 |     21.95 |
| **Fine-Tuned Cross-Encoder (Sectionwise)** | **0.9337** |     **13.64** | **12.47** | **20.89** |

**Key takeaway:**
Task-specific fine-tuning and explicit modeling of document structure (text vs. table) are critical for high-quality retrieval, even under tight latency budgets.

---

### 3.2 Generator Fine-Tuning Results

**Metrics:**

* **Numerical Accuracy**: final answer correctness
* **DSL Accuracy**: exact-match program correctness

| Model                | Input    | Target      | Num. Acc (%) | DSL Acc (%) | Peak Mem (GB) |
| -------------------- | -------- | ----------- | ------------ | ----------- | ------------- |
| Gemma3-1B            | Gold     | Program     | 66.3         | 62.7        | 18.8          |
| Gemma3-4B (QLoRA)    | Gold     | Program     | 61.2         | 57.0        | 8.5           |
| Qwen2.5-Coder-3B     | All      | Program     | 63.3         | 58.2        | 13.7          |
| **Qwen2.5-Coder-3B** | **Gold** | **Program** | **76.6**     | **72.5**    | **13.6**      |

**Key takeaways:**

* Program (DSL) supervision vastly outperforms direct numeric prediction
* Code-specialized pretraining transfers strongly to DSL synthesis
* Smaller fully fine-tuned models can outperform larger QLoRA models

---

### 3.3 Agent Results (Tool-Orchestrated)

**Generator fixed to Qwen2.5-Coder-3B**

| Generator Training | Controller      | Program Acc (%) | Agent Acc (%) |
| ------------------ | --------------- | --------------: | ------------: |
| Noisy-Gold         | Qwen2.5-3B      |            36.4 |          23.4 |
| Gold               | Qwen2.5-3B      |            37.9 |          25.8 |
| Noisy-Gold         | Qwen2.5-14B     |            46.4 |          48.1 |
| **Gold**           | **Qwen2.5-14B** |        **48.8** |      **50.3** |

**Key takeaways:**

* Scaling the **controller**, not the generator, drives agent gains
* Agent accuracy can exceed program exact-match via execution-guided repair
* Tool orchestration has a clear capacity threshold

---

## 4. Reproducibility Instructions

### A. Requirements

* Python **3.11.8**
* Ollama with a model installed (e.g., `qwen2.5:3b`)
* CUDA-compatible GPU (run on RTX 3090 GPU)

---

### B. Environment Setup (micromamba)  

We will use `micromamba` (installation guide:
[https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html))

Please also add `conda-forge` for setting up environments.

```bash
micromamba config append channels conda-forge
```

Create a new environment (`[VERSION]` use 3.11.8):

```bash
micromamba create -n [PROJECT_NAME] python=[VERSION]
```

Activate environment:

```bash
micromamba activate [PROJECT_NAME]
```

You will need to install these:

```bash
micromamba install -c conda-forge python=[VERSION]
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

### C. Installing Ollama  

```bash
brew install ollama
ollama serve
```

---

### D. Running the Script (Inference)  

```bash
ollama pull [MODEL]
python -m agent.run_finqa --model [MODEL] --split dev --max-samples [MAX_SAMPLES]
```

---

### E. Usage Examples  

```bash
# Run on 10 samples from dev set
python -m agent.run_finqa --model [MODEL] --split dev --max-samples 10

# Run single query with verbose output
python -m agent.run_finqa --model [MODEL] --single --query-id 0 --verbose

# Run full dev set
python -m agent.run_finqa --model [MODEL] --split dev
```

---

### F. Command-Line Options  

| Option             | Default    | Description                    |
| ------------------ | ---------- | ------------------------------ |
| `--model`          | `qwen3:4b` | Ollama model name              |
| `--split`          | `dev`      | Dataset split (train/dev/test) |
| `--max-samples`    | 0          | Max samples to run (0 = all)   |
| `--single`         | –          | Run single query mode          |
| `--query-id`       | `0`        | Query ID for single mode       |
| `--verbose`        | –          | Show detailed output           |
| `--max-tool-calls` | 10         | Max tool calls per query       |

---

## 5. Output  

Results are saved to:

```
__output__/exp_<timestamp>/
```

* `config.json` — Experiment configuration
* `results.json` — Full results

---

## 6. Project Structure  

```
.
├── README.md
├── agent
│   ├── dsl
│   ├── retriever
│   ├── orchestrator.py
│   └── run_finqa.py
├── data
├── experiments
└── requirements.txt
```

---

## 7. WandB Dashboards  

* [https://wandb.ai/pgiouroukis-semantic-scholar/finqa-gemma](https://wandb.ai/pgiouroukis-semantic-scholar/finqa-gemma)
* [https://wandb.ai/svk2118-columbia-university/retriever-hpml](https://wandb.ai/svk2118-columbia-university/retriever-hpml)
* [https://wandb.ai/wm2544-columbia-university/](https://wandb.ai/wm2544-columbia-university/)

---