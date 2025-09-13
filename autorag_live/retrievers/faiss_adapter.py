from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os

# Try to import FAISS; fall back to numpy-based implementation if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class DenseRetriever:
    """Base class for dense retrievers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings. Override in subclasses."""
        # Simple deterministic fallback
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            embedding = np.random.randn(384)  # Standard embedding dimension
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents."""
        self.documents = documents
        self.embeddings = self.encode(documents)
        
        if FAISS_AVAILABLE and self.embeddings is not None:
            # Use FAISS for efficient search
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
        else:
            # Fallback to numpy-based search
            self.index = "numpy"
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.encode([query])
        
        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            # FAISS search
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            return results
        else:
            # Numpy fallback
            if self.embeddings is not None:
                # Cosine similarity
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                similarities = np.dot(doc_norms, query_norm.T).flatten()
                
                # Get top k
                top_indices = np.argsort(similarities)[-k:][::-1]
                results = []
                for idx in top_indices:
                    results.append((self.documents[idx], float(similarities[idx])))
                return results
            else:
                return []


class SentenceTransformerRetriever(DenseRetriever):
    """Dense retriever using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = None
        
        # Try to load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("sentence-transformers not available. Using deterministic fallback.")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformers or fallback."""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Deterministic fallback
            return super().encode(texts)


def create_dense_retriever(retriever_type: str = "sentence-transformer", **kwargs) -> DenseRetriever:
    """Factory function for dense retrievers."""
    if retriever_type == "sentence-transformer":
        return SentenceTransformerRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def save_retriever_index(retriever: DenseRetriever, path: str) -> None:
    """Save retriever index and documents."""
    os.makedirs(path, exist_ok=True)
    
    # Save documents
    with open(os.path.join(path, "documents.txt"), "w") as f:
        for doc in retriever.documents:
            f.write(doc + "\n")
    
    # Save embeddings if available
    if retriever.embeddings is not None:
        np.save(os.path.join(path, "embeddings.npy"), retriever.embeddings)
    
    # Save FAISS index if available
    if FAISS_AVAILABLE and isinstance(retriever.index, faiss.Index):
        faiss.write_index(retriever.index, os.path.join(path, "faiss.index"))
    
    # Save config
    config = {
        "model_name": retriever.model_name,
        "retriever_type": retriever.__class__.__name__,
        "num_documents": len(retriever.documents),
        "faiss_available": FAISS_AVAILABLE
    }
    import json
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_retriever_index(path: str) -> DenseRetriever:
    """Load retriever index and documents."""
    import json
    
    # Load config
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)
    
    # Create retriever
    retriever = create_dense_retriever(
        retriever_type="sentence-transformer",
        model_name=config["model_name"]
    )
    
    # Load documents
    with open(os.path.join(path, "documents.txt"), "r") as f:
        retriever.documents = [line.strip() for line in f if line.strip()]
    
    # Load embeddings
    embeddings_path = os.path.join(path, "embeddings.npy")
    if os.path.exists(embeddings_path):
        retriever.embeddings = np.load(embeddings_path)
    
    # Load FAISS index
    faiss_path = os.path.join(path, "faiss.index")
    if FAISS_AVAILABLE and os.path.exists(faiss_path):
        retriever.index = faiss.read_index(faiss_path)
    else:
        retriever.index = "numpy"
    
    return retriever