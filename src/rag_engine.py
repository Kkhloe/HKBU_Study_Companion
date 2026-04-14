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

from utils import load_index, save_index

# Configuration
EMBED_MODEL = "nomic-embed-text"
CHUNKS_DIR = Path("data")      
VECTOR_DB_PATH = "vector_db/index.json"    


# ============================================================
# Part 1: Load pre-chunked files 
# ============================================================

def load_chunks(chunk_file: str) -> List[Dict]:
    file_path = CHUNKS_DIR / chunk_file
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
    raise FileNotFoundError(f"Chunk file not found: {chunk_file} (checked data/)")


# ============================================================
# Part 2: Lexical Retriever
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
    
    def retrieve(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None) -> List[Tuple[float, List[str], Dict]]:
        query_tokens = tokenize(query)
        query_counter = Counter(query_tokens)
        
        # Extract course code from query - support "COMP7810" or "COMP 7810" format
        course_code_match = re.search(r'\b(COMP\s*\d+)\b', query.upper())
        query_course_code = course_code_match.group(1).replace(' ', '') if course_code_match else None
        
        # Combine with explicitly provided course codes
        target_course_codes = set()
        if query_course_code:
            target_course_codes.add(query_course_code)
        if course_codes:
            for code in course_codes:
                # Handle both "COMP7045" and "COMP 7045" formats
                code_match = re.search(r'\b(COMP\s*\d+)\b', code.upper())
                if code_match:
                    # Normalize to COMP#### format (remove spaces)
                    normalized = code_match.group(1).replace(' ', '')
                    target_course_codes.add(normalized)
        
        scored_results = []
        
        for chunk_info in self.chunk_tokens:
            overlap = set(query_counter.keys()) & set(chunk_info["counter"].keys())
            if not overlap:
                continue
            
            # Base score from keyword overlap
            score = sum(
                min(query_counter[word], chunk_info["counter"][word]) 
                for word in overlap
            )
            
            # BOOST SCORE if ANY of the target course codes match
            source_path = chunk_info["metadata"].get("source_path", "")
            for course_code in target_course_codes:
                if course_code in source_path.upper():
                    score += 100  # Strong boost for course code match
                    break  # Avoid multiple boosts for same chunk
            
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
    
    def answer_with_context(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None) -> Tuple[str, List[Dict]]:
        results = self.retrieve(query, top_k, course_codes=course_codes)
        
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
# Part 3: Neural Retriever
# ============================================================

class NeuralRetriever:
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray = None):
        self.chunks = chunks
        self.embeddings = embeddings 

        if self.embeddings is None:
            print(f"Generating embeddings for {len(chunks)} chunks...")
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

    def retrieve(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None) -> List[Tuple[float, Dict]]:
        query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
        
        # Extract course code from query - support "COMP7810" or "COMP 7810" format
        course_code_match = re.search(r'\b(COMP\s*\d+)\b', query.upper())
        query_course_code = course_code_match.group(1).replace(' ', '') if course_code_match else None
        
        # Combine with explicitly provided course codes 
        target_course_codes = set()
        if query_course_code:
            target_course_codes.add(query_course_code)
        if course_codes:
            for code in course_codes:
                # Handle both "COMP7045" and "COMP 7045" formats
                code_match = re.search(r'\b(COMP\s*\d+)\b', code.upper())
                if code_match:
                    # Normalize to COMP#### format (remove spaces)
                    normalized = code_match.group(1).replace(' ', '')
                    target_course_codes.add(normalized)
        
        scored_results = []

        for i, emb_info in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_emb, emb_info)
            
            # BOOST SIMILARITY if ANY of the target course codes match
            source_path = self.chunks[i]["metadata"].get("source_path", "")
            for course_code in target_course_codes:
                if course_code in source_path.upper():
                    similarity += 0.3  # Significant boost for course code match
                    similarity = min(similarity, 1.0)  # Cap at 1.0
                    break  # Avoid multiple boosts for same chunk
            
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

    def answer_with_context(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None) -> Tuple[str, List[Dict]]:
        results = self.retrieve(query, top_k, course_codes=course_codes)
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
# Part 4: RAG Engine
# ============================================================

class RAGEngine:
    def __init__(self, chunk_file: str = "chunks_natural_500_50.jsonl"):
        print(f"Initializing RAG Engine with {chunk_file}")

        self.chunks, self.embeddings = load_index(VECTOR_DB_PATH)

        if self.chunks is None or self.embeddings is None:
            print("No cache found, starting to generate vectors...")
            self.chunks = load_chunks(chunk_file)
            self.embeddings = self._generate_embeddings(self.chunks)
            save_index(self.chunks, self.embeddings, VECTOR_DB_PATH)

        print("Initializing Lexical Retriever...")
        self.lexical = LexicalRetriever(self.chunks)
        
        print("Initializing Neural Retriever (using cache)...")
        self.neural = NeuralRetriever(self.chunks, embeddings=self.embeddings)
        
        print(f"RAG Engine ready! ({len(self.chunks)} chunks)")

    def _generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        print(f"In the generated vector: {len(chunks)} blocks")
        embeddings = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"][:2000]
            emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
            embeddings.append(emb)
            if (i+1) % 50 == 0:
                print(f"   Processed {i+1}/{len(chunks)}")
        return np.array(embeddings)

    # ============================================================
    # Course Code Enhancement Methods 
    # ============================================================
    
    def _extract_course_codes(self, text: str) -> List[str]:
        # Match COMP followed by 4 digits, with optional space
        pattern = r'\b(COMP\s*\d{4})\b'
        matches = re.findall(pattern, text.upper())
        # Normalize: remove spaces, ensure COMP prefix
        return [m.replace(' ', '') for m in matches]
    
    def _enhance_query_with_course_codes(self, query: str, course_codes: List[str]) -> str:
        if not course_codes:
            return query
        
        enhanced = query
        for code in course_codes:
            code_num = code[4:]  # Extract "7045" from "COMP7045"
            with_space = f"COMP {code_num}"
            # Add different formats to improve recall
            enhanced += f" {code} {with_space} {code_num}"
        
        return enhanced
    
    def _filter_chunks_by_course_codes(self, chunks: List[Dict], target_codes: List[str]) -> List[Dict]:
        if not target_codes:
            return chunks
        
        filtered = []
        for chunk in chunks:
            source = chunk["metadata"].get("source_path", "").upper()
            for code in target_codes:
                if code.upper() in source:
                    filtered.append(chunk)
                    break
        return filtered
    
    def _reformat_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant information found."
        
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source_path", "Unknown source")
            page = chunk["metadata"].get("page_label", "?")
            parts.append(f"[{i}] Source: {source} (Page {page})\nContent: {chunk['text'][:500]}...")
        return "\n\n".join(parts)

    # ============================================================
    # Public Search Methods with Course Code Enhancement
    # ============================================================
    
    def lexical_search(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None, 
                       enable_post_filter: bool = False) -> Tuple[str, List[Dict]]:
        # Step 1: Extract course codes from query
        extracted_codes = self._extract_course_codes(query)
        
        # Step 2: Merge with explicitly provided codes
        all_codes = list(set((course_codes or []) + extracted_codes))
        
        # Step 3: Enhance query with course code variations
        enhanced_query = self._enhance_query_with_course_codes(query, all_codes)
        
        # Step 4: Perform retrieval
        context, chunks = self.lexical.answer_with_context(enhanced_query, top_k, course_codes=all_codes)
        
        # Step 5: Optional aggressive post-filtering
        if enable_post_filter and all_codes:
            chunks = self._filter_chunks_by_course_codes(chunks, all_codes)
            context = self._reformat_context(chunks)
        
        return context, chunks
    
    def neural_search(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None,
                      enable_post_filter: bool = False) -> Tuple[str, List[Dict]]:
        # Step 1: Extract course codes from query
        extracted_codes = self._extract_course_codes(query)
        
        # Step 2: Merge with explicitly provided codes
        all_codes = list(set((course_codes or []) + extracted_codes))
        
        # Step 3: Enhance query with course code variations
        enhanced_query = self._enhance_query_with_course_codes(query, all_codes)
        
        # Step 4: Perform retrieval
        context, chunks = self.neural.answer_with_context(enhanced_query, top_k, course_codes=all_codes)
        
        # Step 5: Optional aggressive post-filtering
        if enable_post_filter and all_codes:
            chunks = self._filter_chunks_by_course_codes(chunks, all_codes)
            context = self._reformat_context(chunks)
        
        return context, chunks
    
    def compare_retrievers(self, query: str, top_k: int = 3) -> Dict:
        lexical_context, lexical_chunks = self.lexical_search(query, top_k)
        neural_context, neural_chunks = self.neural_search(query, top_k)
        
        return {
            "query": query,
            "lexical": {
                "context": lexical_context, 
                "chunks": lexical_chunks, 
                "num_chunks": len(lexical_chunks)
            },
            "neural": {
                "context": neural_context, 
                "chunks": neural_chunks, 
                "num_chunks": len(neural_chunks)
            }
        }


# ============================================================
# Test code
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HKBU Study Companion - Enhanced RAG Engine Test")
    print("=" * 70)
    
    engine = RAGEngine(chunk_file="chunks_sliding_500_50.jsonl")
    
    # Test queries with course codes
    test_queries = [
        "COMP7045 grading",
        "7045 assessment",
        "What is the group registration deadline for COMP4146?",
        "COMP 7015 vs COMP 7065 difference"
    ]
    
    for test_query in test_queries:
        print(f"\n" + "=" * 70)
        print(f"Test query: {test_query}")
        print("-" * 70)
        
        print("\nLexical Retrieval (with enhancement):")
        lex_ctx, lex_chunks = engine.lexical_search(test_query, top_k=2)
        print(lex_ctx)
        
        print("\nNeural Retrieval (with enhancement):")
        neu_ctx, neu_chunks = engine.neural_search(test_query, top_k=2)
        print(neu_ctx)
    
    print("\n" + "=" * 70)
    print("Enhancement complete! Ready for integration with chat_logic.py")
    print("=" * 70)