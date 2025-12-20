from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default model path: dsl/model relative to this file
DEFAULT_MODEL_PATH = Path(__file__).parent / "model"

# Singleton instance
_generator: "DSLGenerator | None" = None


def get_generator(model_path: str | None = None) -> "DSLGenerator":
    """
    Get or create the DSL generator singleton.

    Args:
        model_path: Optional path to the model. Uses local dsl/model by default.

    Returns:
        DSLGenerator instance
    """
    global _generator
    if _generator is None:
        path = model_path or str(DEFAULT_MODEL_PATH)
        _generator = DSLGenerator(model_path=path)
    return _generator


def reset_generator() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _generator
    _generator = None


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

