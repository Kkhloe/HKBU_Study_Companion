from pathlib import Path
import json
import numpy as np
from typing import List, Dict

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_index(chunks: List[Dict], embeddings: np.ndarray, save_path: str = "vector_db/index.json"):
    ensure_dir("vector_db")
    data = {
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"The index has been saved: {save_path} ({len(chunks)} blocks)")

def load_index(load_path: str = "vector_db/index.json") -> tuple[List[Dict], np.ndarray]:
    if not Path(load_path).exists():
        return None, None
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"])
    print(f"Index loaded: {len(chunks)} blocks")
    return chunks, embeddings