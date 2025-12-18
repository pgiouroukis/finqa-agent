"""
FinQA Agent package.

Provides an agentic system for answering financial questions using:
- Evidence retrieval (CrossEncoder)
- DSL program generation (Qwen2-5-Coder-3B)
- DSL execution
"""

from .orchestrator import FinQAOrchestrator
from .finqa_data import load_finqa_split, get_question_data
from .dsl_executor import execute_dsl, program_tokenization, eval_program
from .finqa_tools import EvidenceRetriever, DSLGenerator, get_retriever, get_generator

__all__ = [
    "FinQAOrchestrator",
    "load_finqa_split",
    "get_question_data",
    "execute_dsl",
    "program_tokenization",
    "eval_program",
    "EvidenceRetriever",
    "DSLGenerator",
    "get_retriever",
    "get_generator",
]
