from sentence_transformers import CrossEncoder

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
        self.model_path = model_path
        self.model = CrossEncoder(model_path)
        print(f"[EvidenceRetriever] Loaded CrossEncoder from {model_path}")
    
    def retrieve(
        self, 
        question: str,
        evidence_dict: dict[str, str], 
        top_k: int = 3
    ):
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
