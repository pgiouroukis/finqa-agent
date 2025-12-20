"""
FinQA DSL Executor — W&B Profiling Version
(Profile on full FinQA dev.json using gold programs)
"""

from __future__ import annotations
import re
import time
import json
from pathlib import Path
from typing import List, Sequence, Dict, Any, Tuple

import wandb

DEV_JSON_PATH = "data/dev.json"
WANDB_PROJECT = "finqa-executor-profile"

# ============================
# Supported DSL Operations
# ============================
ALL_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
]


# ============================
# Tokenization
# ============================
def program_tokenization(program: str) -> List[str]:
    program = str(program).strip()
    if not program:
        return ["EOF"]

    tokens = []
    for tok in re.split(r",\s*", program):
        cur = ""
        for ch in tok:
            if ch == ")":
                if cur:
                    tokens.append(cur)
                    cur = ""
            cur += ch
            if ch in {"(", ")"}:
                tokens.append(cur)
                cur = ""
        if cur:
            tokens.append(cur)
    tokens.append("EOF")
    return tokens


# ============================
# Utilities
# ============================
def str_to_num(text: str):
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            return float(text.replace("%", "")) / 100.0
        if "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            return float(text)
        return "n/a"


def process_row(row: Sequence[str]):
    out = []
    for cell in row:
        cell = cell.replace("$", "").split("(")[0].strip()
        val = str_to_num(cell)
        if val == "n/a":
            return "n/a"
        out.append(val)
    return out


# ============================
# Core Executor
# ============================
def eval_program(tokens: Sequence[str], table=None) -> Tuple[int, Any]:
    try:
        tokens = list(tokens[:-1])  # drop EOF
        res = {}
        last = "n/a"

        for i, t in enumerate(tokens):
            if i % 4 == 0 and t.strip("(") not in ALL_OPS:
                return 1, "n/a"
            if (i + 1) % 4 == 0 and t != ")":
                return 1, "n/a"

        steps = "|".join(tokens).split(")")[:-1]

        for idx, step in enumerate(steps):
            op = step.split("(")[0].strip("|")
            args = step.split("(")[1].strip("|")
            a1, a2 = [x.strip() for x in args.split("|")]

            def resolve(a):
                if "#" in a:
                    return res.get(int(a.replace("#", "")), "n/a")
                return str_to_num(a)

            if op in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                x, y = resolve(a1), resolve(a2)
                if "n/a" in (x, y):
                    return 1, "n/a"

                if op == "add":
                    last = x + y
                elif op == "subtract":
                    last = x - y
                elif op == "multiply":
                    last = x * y
                elif op == "divide":
                    if y == 0:
                        return 1, "n/a"
                    last = x / y
                elif op == "exp":
                    last = x ** y
                elif op == "greater":
                    last = "yes" if x > y else "no"

            elif "table" in op and table:
                table_map = {r[0]: r[1:] for r in table}
                row = process_row(table_map.get(a1, []))
                if row == "n/a":
                    return 1, "n/a"

                if op == "table_max":
                    last = max(row)
                elif op == "table_min":
                    last = min(row)
                elif op == "table_sum":
                    last = sum(row)
                elif op == "table_average":
                    last = sum(row) / len(row)

            else:
                return 1, "n/a"

            res[idx] = last

        if last not in {"yes", "no", "n/a"}:
            last = round(last, 5)

        return 0, last

    except Exception:
        return 1, "n/a"


# ============================
# Profiling Wrapper
# ============================
def profile_execution(
    program: str,
    gold_value,
    table=None,
):
    tokens = program_tokenization(program)
    num_steps = program.count(")")

    start = time.perf_counter()
    invalid, result = eval_program(tokens, table)
    latency_ms = (time.perf_counter() - start) * 1000

    execution_success = invalid == 0
    execution_accuracy = execution_success and result == gold_value

    error_type = "none"
    if not execution_success:
        error_type = "invalid_or_runtime_error"
    elif execution_success and not execution_accuracy:
        error_type = "wrong_numerical_result"

    wandb.log({
        "execution_latency_ms": latency_ms,
        "program_num_steps": num_steps,
        "execution_success": execution_success,
        "execution_accuracy": execution_accuracy,
        "error_type": error_type,
    })


# ============================
# Main (FinQA dev profiling)
# ============================
if __name__ == "__main__":
    wandb.init(
        project=WANDB_PROJECT,
        name="execution-gold-dev",
        config={
            "dataset": "FinQA",
            "split": "dev",
            "program_source": "gold",
        },
    )

    dev_path = Path(DEV_JSON_PATH)
    with dev_path.open("r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} FinQA dev samples")

    for idx, example in enumerate(dataset):
        qa = example["qa"]

        program = qa["program"]          # gold DSL
        gold_value = qa["exe_ans"]        # gold numerical answer
        table = example.get("table", None)

        profile_execution(
            program=program,
            gold_value=gold_value,
            table=table,
        )

        if idx % 100 == 0:
            print(f"[{idx}/{len(dataset)}] processed")

    wandb.finish()
    print("✅ Finished execution profiling on FinQA dev set")
