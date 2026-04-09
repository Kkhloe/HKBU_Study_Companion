"""
RAG Engine for HKBU Study Companion
Responsible for: document loading, indexing, lexical retrieval, neural retrieval
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import ollama
import numpy as np

# Configuration
EMBED_MODEL = "nomic-embed-text"
CHUNKS_DIR = Path("data")          
VECTOR_DB_DIR = Path("vector_db")  # embeddings cache directory
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Part 1: Load pre-chunked files (done by teammate)
# ============================================================

def load_chunks(chunk_file: str) -> List[Dict]:
    for base_dir in [CHUNKS_DIR, Path("output")]:
        file_path = base_dir / chunk_file
        if file_path.exists():
            chunks = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        data = json.loads(line)
                        chunks.append({
                            "id": f"{chunk_file}_{idx}",
                            "text": data["page_content"],
                            "metadata": data["metadata"]
                        })
            print(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
    raise FileNotFoundError(f"Chunk file not found: {chunk_file} (checked data/ and output/)")

# ============================================================
# Part 2: Lexical Retriever (keyword matching)
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "for", "and", "or", "in", "on", "at", "with", "by", "about",
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "this", "that", "these", "those", "it", "they", "we", "you", "he", "she",
    "it", "them", "us", "him", "her", "its", "their", "our", "your"
}

def tokenize(text: str) -> List[str]:
    """Split text into tokens for lexical retrieval"""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class LexicalRetriever:
    """
    Lexical Retriever: based on keyword matching (like Ctrl+F)
    """
    
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.chunk_tokens = []
        
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            self.chunk_tokens.append({
                "id": chunk["id"],
                "tokens": tokens,
                "counter": Counter(tokens),
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, List[str], Dict]]:
        """
        Retrieve chunks matching the query
        
        Returns:
            List of (score, matched_keywords, chunk_dict)
        """
        query_tokens = tokenize(query)
        query_counter = Counter(query_tokens)
        
        scored_results = []
        
        for chunk_info in self.chunk_tokens:
            overlap = set(query_counter.keys()) & set(chunk_info["counter"].keys())
            if not overlap:
                continue
            
            score = sum(
                min(query_counter[word], chunk_info["counter"][word]) 
                for word in overlap
            )
            
            scored_results.append((
                score, 
                sorted(overlap), 
                {
                    "id": chunk_info["id"],
                    "text": chunk_info["text"],
                    "metadata": chunk_info["metadata"]
                }
            ))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:top_k]
    
    def answer_with_context(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Return formatted context string and retrieved chunks"""
        results = self.retrieve(query, top_k)
        
        if not results:
            return "No relevant information found.", []
        
        context_parts = []
        retrieved_chunks = []
        
        for i, (score, keywords, chunk) in enumerate(results, 1):
            source = chunk["metadata"].get("source_path", "Unknown source")
            page = chunk["metadata"].get("page_label", "?")
            
            context_parts.append(
                f"[{i}] Source: {source} (Page {page})\n"
                f"Content: {chunk['text'][:500]}..."
            )
            retrieved_chunks.append(chunk)
        
        context_str = "\n\n".join(context_parts)
        return context_str, retrieved_chunks


# ============================================================
# Part 3: Neural Retriever (embedding-based)
# ============================================================

class NeuralRetriever:
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray = None):
        self.chunks = chunks
        self.embeddings = embeddings 

        if self.embeddings is None:
            print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
            self.embeddings = []
            for i, chunk in enumerate(chunks):
                text = chunk["text"][:2000]
                emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
                self.embeddings.append(emb)
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i+1}/{len(chunks)}")
            self.embeddings = np.array(self.embeddings)
            print("Embedding generation complete!")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
        scored_results = []

        for i, emb_info in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_emb, emb_info)
            scored_results.append((
                similarity,
                {
                    "id": self.chunks[i]["id"],
                    "text": self.chunks[i]["text"],
                    "metadata": self.chunks[i]["metadata"]
                }
            ))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:top_k]

    def answer_with_context(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant information found.", []

        context_parts = []
        retrieved_chunks = []

        for i, (score, chunk) in enumerate(results, 1):
            meta = chunk["metadata"]
            source = meta.get("source_path", meta.get("title", "Unknown"))
            page = meta.get("page_start") or meta.get("page_label") or meta.get("page", "?")
            doc_type = meta.get("doc_type", "document")
            
            context_parts.append(
                f"[{i}] [{doc_type.upper()}] {source} (Page {page}) | Sim: {score:.4f}\n"
                f"Content: {chunk['text'][:600]}..."
            )
            retrieved_chunks.append(chunk)

        return "\n\n".join(context_parts), retrieved_chunks


# ============================================================
# Part 4: RAG Engine (unified interface)
# ============================================================
class RAGEngine:
    def __init__(self, chunk_file: str = "chunks_natural_500_50.jsonl"):
        print(f" Initializing RAG Engine with {chunk_file}")
        
        self.chunks = load_chunks(chunk_file)
        self.chunk_file = chunk_file

        self.embedding_cache = self._load_or_build_embeddings()
        
        print("Initializing Lexical Retriever...")
        self.lexical = LexicalRetriever(self.chunks)
        
        print("Initializing Neural Retriever (using cache)...")
        self.neural = NeuralRetriever(self.chunks, embeddings=self.embedding_cache)
        
        print(f"RAG Engine ready! ({len(self.chunks)} chunks)")

    def _load_or_build_embeddings(self) -> np.ndarray:
        """embeddings cache: vector_db/{chunk_file}_embeddings.npy"""
        cache_path = VECTOR_DB_DIR / f"{self.chunk_file.replace('.jsonl', '_embeddings.npy')}"
        
        if cache_path.exists():
            print(f" Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        
        print("No cache found → generating embeddings (this runs only once)...")
        embeddings_list = []
        for i, chunk in enumerate(self.chunks):
            emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"][:2000])["embedding"]
            embeddings_list.append(emb)
            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(self.chunks)}")
        
        embeddings = np.array(embeddings_list)
        np.save(cache_path, embeddings)
        print(f"Embeddings cached to {cache_path}")
        return embeddings

    def lexical_search(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        return self.lexical.answer_with_context(query, top_k)

    def neural_search(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        return self.neural.answer_with_context(query, top_k)

    def compare_retrievers(self, query: str, top_k: int = 3) -> Dict:
        """A comparative experiment interface specifically designed for evaluation.ipynb"""
        lexical_context, lexical_chunks = self.lexical_search(query, top_k)
        neural_context, neural_chunks = self.neural_search(query, top_k)
        return {
            "query": query,
            "lexical": {"context": lexical_context, "chunks": lexical_chunks, "num_chunks": len(lexical_chunks)},
            "neural": {"context": neural_context, "chunks": neural_chunks, "num_chunks": len(neural_chunks)}
        }
# ============================================================
# Test code (runs when this file is executed directly)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HKBU Study Companion - Optimized RAG Engine Test")
    print("=" * 70)
    
    engine = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
    
    test_query = "What is the group registration deadline for COMP4146?"
    
    print(f"\nTest query: {test_query}")
    print("-" * 70)
    
    print("\nLexical Retrieval:")
    lex_ctx, lex_chunks = engine.lexical_search(test_query, top_k=2)
    print(lex_ctx)
    
    print("\nNeural Retrieval:")
    neu_ctx, neu_chunks = engine.neural_search(test_query, top_k=2)
    print(neu_ctx)
    
    print("\n Optimization complete! Ready for direct integration chat_logic.py")
