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
VECTOR_DB_DIR = Path("vector_db")


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


# Lexical Retriever
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "for", "and", "or", "in", "on", "at", "with", "by", "about",
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "this", "that", "these", "those", "it", "they", "we", "you", "he", "she",
    "it", "them", "us", "him", "her", "its", "their", "our", "your"
}

def tokenize(text: str) -> List[str]:
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
        
        target_course_codes = set()
        
        # Match all common course code patterns
        patterns = [
            r'\b(COMP\s*\d{4})\b',
            r'\b(DAAI\s*\d{4})\b',
            r'\b(ITM\s*\d{4})\b',
            r'\b(AIDM\s*\d{4})\b',
            r'\b([A-Z]{3,4}\s*\d{4,6})\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            for m in matches:
                target_course_codes.add(m.replace(' ', ''))
        
        if course_codes:
            for code in course_codes:
                target_course_codes.add(code.upper().replace(' ', ''))
        
        filtered_chunks = []
        for chunk_info in self.chunk_tokens:
            if target_course_codes:
                source_path = chunk_info["metadata"].get("source_path", "").upper()
                keep = False
                for code in target_course_codes:
                    if code in source_path:
                        keep = True
                        break
                if not keep:
                    continue
            filtered_chunks.append(chunk_info)

        
        scored_results = []

        # Search only within the filtered set
        for chunk_info in filtered_chunks:
            overlap = set(query_counter.keys()) & set(chunk_info["counter"].keys())
            if not overlap:
                continue
            
            # Base score from token overlap
            score = sum(min(query_counter[word], chunk_info["counter"][word]) for word in overlap)
            
            # Boost score for exact course match
            source_path = chunk_info["metadata"].get("source_path", "")
            for code in target_course_codes:
                if code in source_path.upper():
                    score += 100
                    break
            
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
        """Return formatted context string and chunks"""
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


# Neural Retriever
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

    def retrieve(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None) -> List[Tuple[float, Dict]]:
        query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
        
        target_course_codes = set()
        
        patterns = [
            r'\b(COMP\s*\d{4})\b',
            r'\b(DAAI\s*\d{4})\b',
            r'\b(ITM\s*\d{4})\b',
            r'\b(AIDM\s*\d{4})\b',
            r'\b([A-Z]{3,4}\s*\d{4,6})\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            for m in matches:
                target_course_codes.add(m.replace(' ', ''))
        
        if course_codes:
            for code in course_codes:
                target_course_codes.add(code.upper().replace(' ', ''))
        
        filtered_indices = []
        for i, chunk in enumerate(self.chunks):
            if target_course_codes:
                source_path = chunk["metadata"].get("source_path", "").upper()
                keep = False
                for code in target_course_codes:
                    if code in source_path:
                        keep = True
                        break
                if not keep:
                    continue
            filtered_indices.append(i)

        
        scored_results = []

        # Compute similarity only within the filtered set
        for i in filtered_indices:
            emb_info = self.embeddings[i]
            chunk = self.chunks[i]
            
            similarity = self._cosine_similarity(query_emb, emb_info)
            
            # Boost similarity for course match
            source_path = chunk["metadata"].get("source_path", "")
            for code in target_course_codes:
                if code in source_path.upper():
                    similarity += 0.3
                    similarity = min(similarity, 1.0)
                    break
            
            scored_results.append((
                similarity,
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
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


# RAG Engine
class RAGEngine:
    def __init__(self, chunk_file: str = "chunks_natural_500_50.jsonl"):
        print(f"Initializing RAG Engine with {chunk_file}")
        
        self.chunks = load_chunks(chunk_file)
        self.chunk_file = chunk_file
        self.embedding_cache = self._load_or_build_embeddings()
        
        print("Initializing Lexical Retriever...")
        self.lexical = LexicalRetriever(self.chunks)
        
        print("Initializing Neural Retriever (using cache)...")
        self.neural = NeuralRetriever(self.chunks, embeddings=self.embedding_cache)
        
        print(f"RAG Engine ready! ({len(self.chunks)} chunks)")

    def _load_or_build_embeddings(self) -> np.ndarray:
        """Load cached embeddings or generate new ones"""
        cache_path = VECTOR_DB_DIR / f"{self.chunk_file.replace('.jsonl', '_embeddings.npy')}"
        
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        
        print("No cache found → generating embeddings...")
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

    # Course Code Enhancement Methods
    def _extract_course_codes(self, text: str) -> List[str]:
        """Extract course codes from text"""
        patterns = [
            r'\b(COMP\s*\d{4})\b',
            r'\b([A-Z]{3,4}\s*\d{4,6})\b',
        ]
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            all_matches.extend(matches)
        return list(set([m.replace(' ', '') for m in all_matches]))
    
    def _enhance_query_with_course_codes(self, query: str, course_codes: List[str]) -> str:
        """Enhance query with course code variations"""
        if not course_codes:
            return query
        
        enhanced = query
        for code in course_codes:
            code_num = re.sub(r'[A-Z]+', '', code)
            with_space = f"{code[:4]} {code_num}" if len(code) > 4 else code
            enhanced += f" {code} {with_space}"
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
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                unique_chunks.append(chunk)
        return unique_chunks
    
    def _reformat_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant information found."
        
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source_path", "Unknown source")
            page = chunk["metadata"].get("page_label", "?")
            parts.append(f"[{i}] Source: {source} (Page {page})\nContent: {chunk['text'][:500]}...")
        return "\n\n".join(parts)

    # Public Search Methods
    def lexical_search(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None, 
                       enable_post_filter: bool = False, deduplicate: bool = True) -> Tuple[str, List[Dict]]:
        extracted_codes = self._extract_course_codes(query)
        # If caller explicitly locks course codes, do NOT expand with codes found in prompt text.
        # This prevents prompt templates (e.g., examples mentioning other courses) from polluting retrieval.
        if course_codes:
            all_codes = list(dict.fromkeys([c.upper().replace(" ", "") for c in course_codes if c]))
        else:
            all_codes = extracted_codes
        enhanced_query = self._enhance_query_with_course_codes(query, all_codes)
        
        fetch_k = top_k * 2 if deduplicate else top_k
        context, chunks = self.lexical.answer_with_context(enhanced_query, fetch_k, course_codes=all_codes)
        
        if enable_post_filter and all_codes:
            chunks = self._filter_chunks_by_course_codes(chunks, all_codes)
        
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
            chunks = chunks[:top_k]
            context = self._reformat_context(chunks)
        
        return context, chunks
    
    def neural_search(self, query: str, top_k: int = 3, course_codes: Optional[List[str]] = None,
                      enable_post_filter: bool = False, deduplicate: bool = True) -> Tuple[str, List[Dict]]:
        extracted_codes = self._extract_course_codes(query)
        # If caller explicitly locks course codes, do NOT expand with codes found in prompt text.
        if course_codes:
            all_codes = list(dict.fromkeys([c.upper().replace(" ", "") for c in course_codes if c]))
        else:
            all_codes = extracted_codes
        enhanced_query = self._enhance_query_with_course_codes(query, all_codes)
        
        fetch_k = top_k * 2 if deduplicate else top_k
        context, chunks = self.neural.answer_with_context(enhanced_query, fetch_k, course_codes=all_codes)
        
        if enable_post_filter and all_codes:
            chunks = self._filter_chunks_by_course_codes(chunks, all_codes)
        
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
            chunks = chunks[:top_k]
            context = self._reformat_context(chunks)
        
        return context, chunks
    
    def compare_retrievers(self, query: str, top_k: int = 3) -> Dict:
        lexical_context, lexical_chunks = self.lexical_search(query, top_k)
        neural_context, neural_chunks = self.neural_search(query, top_k)
        
        return {
            "query": query,
            "lexical": {"context": lexical_context, "chunks": lexical_chunks, "num_chunks": len(lexical_chunks)},
            "neural": {"context": neural_context, "chunks": neural_chunks, "num_chunks": len(neural_chunks)}
        }


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("HKBU Study Companion - RAG Engine Test")
    print("=" * 70)
    
    engine = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
    
    test_queries = [
        "COMP7980 final exam",
        "COMP7530 assessment", 
        "What is the policy for COMP7055?",
    ]
    
    for test_query in test_queries:
        print(f"\n" + "=" * 70)
        print(f"Test query: {test_query}")
        print("-" * 70)
        
        print("\nLexical Retrieval:")
        lex_ctx, lex_chunks = engine.lexical_search(test_query, top_k=2)
        print(lex_ctx)
        
        print("\nNeural Retrieval:")
        neu_ctx, neu_chunks = engine.neural_search(test_query, top_k=2)
        print(neu_ctx)
    
    print("\n✅ RAG Engine test completed.")