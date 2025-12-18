"""
DSL Executor for FinQA.

Ported from finetune_gemma (1).py - handles parsing and execution of 
FinQA Domain Specific Language programs.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Dict, Any, Tuple


# All supported FinQA DSL operations
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


def program_tokenization(original_program: str) -> List[str]:
    """
    Tokenize a FinQA DSL program string into tokens.
    
    Example: "subtract(100, 50), divide(#0, 2)" -> 
             ["subtract(", "100", "50", ")", "divide(", "#0", "2", ")", "EOF"]
    """
    if not isinstance(original_program, str):
        original_program = str(original_program)
    original_program = original_program.strip()
    if not original_program:
        return ["EOF"]
    
    program: List[str] = []
    for tok in re.split(r",\s*", original_program):
        if not tok:
            continue
        cur_tok = ""
        for char in tok:
            if char == ")":
                if cur_tok:
                    program.append(cur_tok)
                    cur_tok = ""
            cur_tok += char
            if char in {"(", ")"}:
                program.append(cur_tok)
                cur_tok = ""
        if cur_tok:
            program.append(cur_tok)
    program.append("EOF")
    return program


def str_to_num(text: str):
    """Convert a string to a number, handling percentages and constants."""
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text) / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def process_row(row_in: Sequence[str]):
    """Process a table row, converting all cells to numbers."""
    row_out: List[float] = []
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        num_value = str_to_num(num)
        if num_value == "n/a":
            return "n/a"
        row_out.append(num_value)
    return row_out


def eval_program(
    program: Sequence[str], 
    table: Sequence[Sequence[str]] | None
) -> Tuple[int, Any]:
    """
    Execute a tokenized FinQA DSL program.
    
    Args:
        program: Tokenized program (from program_tokenization)
        table: Optional table data for table operations
    
    Returns:
        Tuple of (invalid_flag, result)
        - invalid_flag: 0 if successful, 1 if error
        - result: computed result or "n/a"
    """
    invalid_flag = 0
    this_res = "n/a"
    
    try:
        program = list(program[:-1])  # remove EOF
        
        # Validate program structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if op in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                # Resolve arg1
                if "#" in arg1:
                    arg1_val = res_dict.get(int(arg1.replace("#", "")), "n/a")
                else:
                    arg1_val = str_to_num(arg1)
                if arg1_val == "n/a":
                    invalid_flag = 1
                    break

                # Resolve arg2
                if "#" in arg2:
                    arg2_val = res_dict.get(int(arg2.replace("#", "")), "n/a")
                else:
                    arg2_val = str_to_num(arg2)
                if arg2_val == "n/a":
                    invalid_flag = 1
                    break

                # Execute operation
                if op == "add":
                    this_res = arg1_val + arg2_val
                elif op == "subtract":
                    this_res = arg1_val - arg2_val
                elif op == "multiply":
                    this_res = arg1_val * arg2_val
                elif op == "divide":
                    this_res = arg1_val / arg2_val
                elif op == "exp":
                    this_res = arg1_val ** arg2_val
                elif op == "greater":
                    this_res = "yes" if arg1_val > arg2_val else "no"
                res_dict[ind] = this_res
                
            elif "table" in op and table is not None:
                table_dict = {row[0]: row[1:] for row in table}
                if "#" in arg1:
                    arg1_val = res_dict.get(int(arg1.replace("#", "")), "n/a")
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    num_row = process_row(table_dict[arg1])
                    arg1_val = num_row

                if arg1_val == "n/a":
                    invalid_flag = 1
                    break
                    
                if op == "table_max":
                    this_res = max(arg1_val)
                elif op == "table_min":
                    this_res = min(arg1_val)
                elif op == "table_sum":
                    this_res = sum(arg1_val)
                elif op == "table_average":
                    this_res = sum(arg1_val) / len(arg1_val)
                res_dict[ind] = this_res
            else:
                invalid_flag = 1
                break

        if this_res not in {"yes", "no", "n/a"}:
            this_res = round(this_res, 5)
            
    except Exception:
        invalid_flag = 1
        
    return invalid_flag, this_res


def execute_dsl(program_str: str, table: Sequence[Sequence[str]] | None = None) -> Dict[str, Any]:
    """
    High-level DSL execution function for the agent.
    
    Args:
        program_str: Raw DSL program string
        table: Optional table data
    
    Returns:
        Dict with execution results
    """
    tokens = program_tokenization(program_str)
    invalid_flag, result = eval_program(tokens, table or [])
    
    return {
        "program": program_str,
        "tokens": tokens,
        "success": invalid_flag == 0,
        "result": result,
        "error": "Invalid program or execution error" if invalid_flag else None,
    }


def clean_prediction_text(text: str, eos_token: str | None = None) -> str:
    """Clean up generated DSL prediction text."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:", 1)[-1].strip()
    if "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[0].strip()
    if eos_token and eos_token in cleaned:
        cleaned = cleaned.split(eos_token, 1)[0].strip()
    return cleaned
