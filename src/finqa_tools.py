"""
FinQA Tools for the Agent.

Implements the tools the agent can use:
1. retrieve_evidence - CrossEncoder reranker for evidence retrieval
2. generate_dsl - Qwen2-5-Coder-3B for DSL program generation
3. execute_dsl - DSL interpreter
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch


# =========================================
# Evidence Retriever (CrossEncoder)
# =========================================

class EvidenceRetriever:
    """
    Evidence retriever using a trained CrossEncoder reranker.
    
    Scores all evidence pieces against the question and returns top-k.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the retriever.
        
        Args:
            model_path: Path to the trained CrossEncoder checkpoint
        """
        from sentence_transformers import CrossEncoder
        
        self.model_path = model_path
        self.model = CrossEncoder(model_path)
        print(f"[EvidenceRetriever] Loaded CrossEncoder from {model_path}")
    
    def retrieve(
        self, 
        question: str, 
        evidence_dict: Dict[str, str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k evidence pieces for a question.
        
        Args:
            question: The question to retrieve evidence for
            evidence_dict: Dict mapping evidence_id -> evidence_text
            top_k: Number of top results to return
        
        Returns:
            List of dicts with evidence_id, text, and score
        """
        if not evidence_dict:
            return []
        
        # Create pairs for scoring
        keys = list(evidence_dict.keys())
        pairs = [[question, evidence_dict[k]] for k in keys]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Rank and return top-k
        ranked = sorted(zip(keys, scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for evidence_id, score in ranked[:top_k]:
            results.append({
                "evidence_id": evidence_id,
                "text": evidence_dict[evidence_id],
                "score": float(score),
            })
        
        return results


# =========================================
# DSL Generator (Qwen2-5-Coder-3B)
# =========================================

class DSLGenerator:
    """
    DSL program generator using a fine-tuned Qwen2-5-Coder-3B model.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_path = model_path
        self.device = device
        
        print(f"[DSLGenerator] Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device if device != "auto" else "auto",
        )
        self.model.eval()
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[DSLGenerator] Model loaded on {self.model.device}")
    
    def generate(
        self, 
        question: str, 
        evidence: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate a DSL program for the given question and evidence.
        
        Args:
            question: The financial question
            evidence: Retrieved evidence context
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with generated program and metadata
        """
        # Build prompt (same format as training)
        hint = "Generate the FinQA reasoning program (DSL) that answers the question."
        prompt = f"{hint}\n\nQuestion:\n{question}\n\nContext:\n{evidence}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        
        # Clean up the prediction
        if "\n" in generated:
            generated = generated.split("\n")[0].strip()
        
        return {
            "program": generated,
            "prompt_length": inputs["input_ids"].shape[1],
            "generated_length": outputs[0].shape[0] - inputs["input_ids"].shape[1],
        }


# =========================================
# Global Tool Instances (lazy loaded)
# =========================================

_retriever: EvidenceRetriever | None = None
_generator: DSLGenerator | None = None

# Default model paths
DEFAULT_RETRIEVER_PATH = "/home/pg2860/hpml-run/reranker-sweep/jjyfsr7j/checkpoint-1200"
DEFAULT_GENERATOR_PATH = "/home/pg2860/hpml/__output__/qwen2-5-coder-3b-instruct-full-input_gold-output_program/final_model"


def get_retriever(model_path: str | None = None) -> EvidenceRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        path = model_path or DEFAULT_RETRIEVER_PATH
        _retriever = EvidenceRetriever(path)
    return _retriever


def get_generator(model_path: str | None = None) -> DSLGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        path = model_path or DEFAULT_GENERATOR_PATH
        _generator = DSLGenerator(path)
    return _generator


def reset_tools():
    """Reset all tool instances (for testing)."""
    global _retriever, _generator
    _retriever = None
    _generator = None
