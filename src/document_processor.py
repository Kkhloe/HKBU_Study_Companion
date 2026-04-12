"""
RAG Document Pre-processing Script
Loads PDF/TXT → Splits with two methods → Enhances metadata → Outputs JSONL
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Literal

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Utility functions
def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def detect_language_zh_en(text: str) -> str:
    if not text:
        return "en"
    chinese_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    ratio = chinese_count / len(text)
    return "zh" if ratio > 0.30 else "en"

def infer_doc_type(filename: str) -> str:
    name = Path(filename).name.lower()
    rules = [
        ('msc international student guide', 'handbook'),
        ('programme handout', 'handout'),
        ('guidelines', 'guide'),
        ('policy', 'policy'),
        ('strategy', 'policy'),
        (re.compile(r'comp\d+'), 'syllabus'),
    ]
    for pattern, doc_type in rules:
        if isinstance(pattern, re.Pattern):
            if pattern.search(name):
                return doc_type
        elif pattern in name:
            return doc_type
    return 'document'

def get_doc_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    parts = re.split(r'[^a-z0-9]+', stem)
    parts = [p for p in parts if p]
    return f"{parts[0]}_001" if parts else "document_001"

def load_documents(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    project_root = Path(__file__).parent.parent
    docs: List[Document] = []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in ['.pdf', '.txt']]

    print(f"Found {len(files)} document files")
    for file_path in files:
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), autodetect_encoding=True)
            
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)
                meta = dict(doc.metadata) if doc.metadata else {}
                
                meta['source_path'] = str(file_path.resolve())
                meta['title'] = file_path.stem
                meta['doc_type'] = infer_doc_type(str(file_path))
                meta['doc_id'] = get_doc_id_from_filename(str(file_path))
                meta['language'] = detect_language_zh_en(doc.page_content)
                
                if 'page' in meta and isinstance(meta['page'], int):
                    meta['page_start'] = meta['page'] + 1
                    meta['page_end'] = meta['page'] + 1
                else:
                    meta['page_start'] = meta['page_end'] = None
                
                doc.metadata = meta
                docs.append(doc)
            
            print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")
    return docs

def sliding_window(docs: List[Document], chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    print(f"Sliding window chunking (size={chunk_size}, overlap={overlap})")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator="")
    return _split_documents(docs, splitter, "sliding")

def natural(docs: List[Document], chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    print(f"Natural boundary chunking (size={chunk_size}, overlap={overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " ", ""],
    )
    return _split_documents(docs, splitter, "natural")

def _split_documents(docs: List[Document], splitter, method: str) -> List[Document]:
    chunks: List[Document] = []
    for doc in docs:
        doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
        split_docs = splitter.split_documents([doc_copy])
        for i, chunk in enumerate(split_docs):
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata.update(doc.metadata)
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_id'] = f"{doc.metadata.get('doc_id', 'doc')}chunk{i:03d}"
            chunks.append(chunk)
    print(f"{method.capitalize()} splitting complete: {len(chunks)} chunks")
    return chunks


def enhance_metadata(docs: List[Document]) -> List[Document]:
    for doc in docs:
        m = doc.metadata
        m.setdefault('source_path', None)
        m.setdefault('title', 'unknown')
        m.setdefault('doc_type', 'document')
        m.setdefault('doc_id', 'document_001')
        m.setdefault('language', detect_language_zh_en(doc.page_content))
        m.setdefault('chunk_index', 0)
        if 'chunk_id' not in m:
            m['chunk_id'] = f"{m['doc_id']}chunk{m['chunk_index']:03d}"
    return docs

def save_as_jsonl(docs: List[Document], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            obj = {'page_content': doc.page_content, 'metadata': doc.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"Saved {len(docs)} chunks to {output_file}")

def prepare_chunks(folder_path: str, method: Literal["natural", "sliding"] = "natural",
                   chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    print(f"\nDocument preprocessing: method={method}, size={chunk_size}, overlap={overlap}")
    
    docs = load_documents(folder_path)
    if not docs:
        print("Error: No documents loaded")
        return []

    if method == "sliding":
        chunks = sliding_window(docs, chunk_size, overlap)
    else:
        chunks = natural(docs, chunk_size, overlap)

    chunks = enhance_metadata(chunks)
    print(f"Total: {len(chunks)} chunks")
    return chunks


def calculate_boundary_cut_ratio(chunks: List[Document]) -> float:
    if not chunks:
        return 0.0

    mid_punct = set(',，；;')
    end_punct = set('。.!！？」?')
    boundary_cuts = 0

    for chunk in chunks:
        content = chunk.page_content
        if not content:
            continue
        first = content[0]
        last = content[-1]
        length = len(content)

        if first in mid_punct or (length > 50 and last not in end_punct):
            boundary_cuts += 1

    return boundary_cuts / len(chunks)

def compare_chunking_methods(folder_path: str) -> str:
    print("\nComparing chunking methods...")
    original_docs = load_documents(folder_path)
    if not original_docs:
        print("Error: No documents loaded")
        return "natural"

    chunks_sliding = sliding_window(original_docs)
    chunks_sliding = enhance_metadata(chunks_sliding)

    original_docs_2 = load_documents(folder_path)
    chunks_natural = natural(original_docs_2)
    chunks_natural = enhance_metadata(chunks_natural)

    def calc_stats(chunks):
        if not chunks:
            return {'count': 0, 'avg_len': 0, 'min_len': 0, 'max_len': 0}
        lengths = [len(c.page_content) for c in chunks]
        return {
            'count': len(chunks),
            'avg_len': sum(lengths) // len(lengths),
            'min_len': min(lengths),
            'max_len': max(lengths)
        }

    stats_sliding = calc_stats(chunks_sliding)
    stats_natural = calc_stats(chunks_natural)
    ratio_sliding = calculate_boundary_cut_ratio(chunks_sliding)
    ratio_natural = calculate_boundary_cut_ratio(chunks_natural)

    print(f"\n{'Metric':<25} {'Sliding Window':<18} {'Natural Boundary':<18}")
    print("-" * 61)
    print(f"{'Total chunks':<25} {stats_sliding['count']:<18} {stats_natural['count']:<18}")
    print(f"{'Average length':<25} {stats_sliding['avg_len']:<18} {stats_natural['avg_len']:<18}")
    print(f"{'Min length':<25} {stats_sliding['min_len']:<18} {stats_natural['min_len']:<18}")
    print(f"{'Max length':<25} {stats_sliding['max_len']:<18} {stats_natural['max_len']:<18}")
    print(f"{'Boundary-cut ratio':<25} {ratio_sliding:<18.2%} {ratio_natural:<18.2%}")

    recommended = "natural" if ratio_natural <= ratio_sliding else "sliding"
    print(f"\nRecommended: {recommended}")
    return recommended

def main():
    parser = argparse.ArgumentParser(description="HKBU Study Companion - Document preprocessing")
    parser.add_argument('--input', '-i', default='data', help='Input folder (default: data)')
    parser.add_argument('--chunker', '-m', choices=['natural', 'sliding'], default='natural')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=50)
    parser.add_argument('--out', '-o', help='Output JSONL path')
    parser.add_argument('--build-both', action='store_true', help='Generate both chunking methods')
    
    args = parser.parse_args()
    input_path = Path(args.input)

    if args.build_both:
        for method in ["natural", "sliding"]:
            chunks = prepare_chunks(str(input_path), method=method, 
                                  chunk_size=args.chunk_size, overlap=args.overlap)
            out_path = f"data/chunks_{method}_{args.chunk_size}_{args.overlap}.jsonl"
            save_as_jsonl(chunks, out_path)
        return

    if args.out:
        chunks = prepare_chunks(str(input_path), method=args.chunker,
                              chunk_size=args.chunk_size, overlap=args.overlap)
        save_as_jsonl(chunks, args.out)
    else:
        recommended = compare_chunking_methods(str(input_path))

if __name__ == "__main__":
    main()