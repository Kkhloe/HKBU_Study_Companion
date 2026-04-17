# HKBU Study Companion

Local offline-first study assistant for Hong Kong Baptist University, built with **Ollama + RAG** and a clean **Streamlit** UI.

This tool helps HKBU students organize course materials, retrieve lecture knowledge, and generate cited answers & customized study plans with fully local deployment.

## Key Features
- Local knowledge base construction from PDF / TXT course materials
- Dual retrieval: lexical search & embedding-based semantic search
- Context-aware LLM generation with source citations
- Multiple chunking strategies: natural splitting & sliding window
- Intuitive Streamlit web UI + lightweight CLI mode
- Embedding cache for faster repeated queries
- ReAct reasoning loop (Thought-Action-Observation)
- LLM-as-a-Judge automatic answer evaluation

## System Requirements
- Windows 10 (PowerShell)
- Python 3.11.8
- Local Ollama service (running in background)
- Storage: several GB for LLM & embedding cache

## Tech Stack
- Backend: Python
- Local LLM: Ollama
- RAG Core: Custom retrieval & document processing
- Frontend: Streamlit
- Reasoning: ReAct
- Evaluation: LLM-as-a-Judge
- Data Format: JSONL

---

## Quick Start

### 1. Create Virtual Environment & Install Dependencies
Run all commands under **project root directory**:
```powershell
# Create virtual environment
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip

# Install required packages
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Install Ollama & Pull Required Models
1. Download and install [Ollama](https://ollama.com/)
2. Execute in PowerShell:
```powershell
# Check Ollama status
ollama --version

# Required embedding model for RAG
ollama pull nomic-embed-text

# Default lightweight generation model
ollama pull gemma3:4b
```

Optional alternative models:
```powershell
ollama pull gemma3:12b
ollama pull qwen3:8b
```

---

## Prepare Course Data

### 1. Organize Documents
Place all lecture notes, readings and handbooks into the `data/` folder:
```
data/
├─ COMP7045/
├─ COMP7950/
└─ policies/
```

### 2. Generate Chunk Files
Build both natural chunking and sliding window chunking:
```powershell
.\.venv\Scripts\python.exe src\document_processor.py --input data --build-both --chunk-size 500 --overlap 50
```

Generated files:
- `data/chunks_natural_500_50.jsonl`
- `data/chunks_sliding_500_50.jsonl`

---

## Launch Application

### Streamlit Web UI (Recommended)
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

### CLI Demo (No UI)
```powershell
.\.venv\Scripts\python.exe src/main.py
```
Type `exit` to terminate the CLI session.

> Note:
> First launch will generate embedding cache in `/vector_db/`.
> Delete this folder if you need to refresh all vector data.

---

## Extra Dependencies (For Analysis Notebooks)
```powershell
# For visualization & data analysis
.\.venv\Scripts\python.exe -m pip install matplotlib pandas

# Optional: word cloud generation
.\.venv\Scripts\python.exe -m pip install wordcloud
```

---

# Project Modules

## Retriever - Shu Yaming
Path: `HKBUStudyCompanion/src/rag_engine.py`

### Usage
**Install Dependencies:**
```bash
pip install ollama numpy
```

**Download Model:**
```bash
ollama pull nomic-embed-text
```

**Calling the Module:**
```python
from src.rag_engine import RAGEngine

# Initialize (once)
engine = RAGEngine(chunk_file="chunks_sliding_500_50.jsonl")

# Lexical retrieval
context, chunks = engine.lexical_search("your question", top_k=3)

# Neural retrieval
context, chunks = engine.neural_search("your question", top_k=3)

# Compare both retrievers
result = engine.compare_retrievers("your question", top_k=3)
```

---

## Evaluation 
File: `evaluation_update.ipynb`

This notebook is the **main evaluation** artifact. It runs top-to-bottom and produces figures/tables for project reports (quality + token-efficiency evidence).

Aligned with unified interface:
- `rag = RAGEngine(...)`
- `pm = PromptManager()`
- `chat = ChatLogic(rag, pm)`
- `chat.process_query(...)`

### What this notebook covers
- Retrieval sanity check: lexical vs neural
- Baseline: No-RAG vs Neural RAG
- Retriever comparison: Lexical vs Neural
- Context optimization: Raw vs summarized context
- Token-efficiency measurement & top-k tradeoff plots
- Visualization: wordcloud / top-words bar chart

### Prerequisites
- Ollama running
- Models: `nomic-embed-text`, `gemma3:4b`

### Environment Setup (Windows PowerShell)
```powershell
cd "e:\HKBU\COMP7125 Prompt Engineering\lab"
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install ollama numpy matplotlib pandas
```

Optional wordcloud:
```powershell
.\venv\Scripts\python.exe -m pip install wordcloud
```

### Data Requirement
Uses chunk file:
- `data/chunks_natural_500_50.jsonl`

### How to Run
1. Open `notebooks/evaluation_update.ipynb`
2. Run cells top-to-bottom
3. If OOM occurs: reduce `top_k` or use smaller models

### Report Outputs
- Baseline: No-RAG vs RAG
- Retriever: Lexical vs Neural
- Context: Raw vs Summarized
- Visual figures: wordcloud, top-k tradeoff plots

---

## Chat Logic - He Bien

**What it is:**
A module that runs one round of chat: task detection → retrieve → prompt → call Ollama → return answer & stats.

**Main pieces:**
- `ChatLogic(rag_engine, prompt_manager)` — create once
- `process_query(...)` — normal RAG chat
- `process_query_no_rag(...)` — baseline
- `compare_no_rag_vs_rag` / `compare_lexical_vs_neural` — A/B test
- `clear_history()` — reset conversation

**Dependencies:**
`rag_engine.py`, `prompt_manager.py`, Ollama, `data/` chunks

**Used by:**
`app.py`, `main.py`, evaluation notebooks

---

## ReAct & LLM-as-a-Judge - Cai Xueying

### Introduction of ReAct
#### Core Features
- Implements Thought-Action-Observation loop reasoning
- Configurable reasoning steps from UI
- Fully traceable reasoning process
- Supports custom action handlers

#### Added Modules
- `src/react_engine.py` — core ReAct engine
- `src/react_prompt_manager.py` — ReAct prompt optimization

#### Modified Modules
- `src/chat_logic.py` — added ReAct engine & hooks
- `src/app.py` — added ReAct UI panel & controls

#### How to Run
```bash
streamlit run src/app.py
```

#### UI Display
- Sidebar: Enable ReAct & adjust steps
- Expandable reasoning process panel

---

### Introduction of LLM-as-a-Judge

#### Added Modules
- `judge_engine.py` — LLM evaluator core
- Added `judge_response` in ChatLogic
- UI “LLM Judge” button in app.py

#### Purpose
Uses LLM (e.g., gemma3:12b) to automatically evaluate answer quality, output score, reasoning, and suggestions.

#### Main Features
- Automatic 1–5 scoring
- Configurable judge model
- Structured output: `score`, `reasoning`, `suggestion`

#### Usage Example
```python
from LLM_judge_engine import JudgeEngine

judge = JudgeEngine(model="gemma3:12b")
result = judge.judge(query="What is RAG?", answer="RAG means ...", context="...")
print(result)
```

---

## Project Structure
```
src/
├─ app.py               # Streamlit web entry
├─ main.py              # CLI demo entry
├─ rag_engine.py        # Retrieval & embedding cache (Shu Yaming)
├─ document_processor.py# PDF/TXT parsing & chunking
├─ chat_logic.py        # RAG pipeline (He Bien)
├─ prompt_manager.py    # Prompt template & token control
├─ react_engine.py      # ReAct reasoning (Cai Xueying)
├─ react_prompt_manager.py
└─ judge_engine.py      # LLM-as-a-Judge (Cai Xueying)

data/                   # Course materials & chunk files
vector_db/              # Embedding cache
notebooks/
└─ evaluation_update.ipynb  # Evaluation (He Bien)
screenshot/             # UI figures
```

---

## Common Issues & Solutions

### Chunk File Not Found
- Run all commands from **project root**
- Regenerate chunk files with `document_processor.py`

### Ollama Out-of-Memory Error
- Switch to smaller models (e.g., gemma3:4b)
- Lower `top_k` and generation length in UI settings

### Absolute Path in Citation Sources
- Rebuild chunk files to generate relative portable paths
```
