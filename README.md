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
- Budget-aware structured prompt engineering system

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
- Prompt Engine: Budget-controlled PromptManager
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

## Retriever
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

## Chat Logic

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

## Prompt Manager
Path: `src/prompt_manager.py`

### Overview
Prompt Manager is the core prompt engineering module for HKBU Study Companion.
It implements **data-driven prompt composition, token budget control, priority-based element management, and task-adaptive generation configuration**.

Core capabilities：
- Modular & prioritized prompt element design
- Fine-grained token budget allocation and overflow protection
- Automatic context/history truncation
- Four task modes with independent LLM generation parameters
- Standardized system instructions, constraints and response rules
- Full metadata recording for prompt assembly & token consumption

### Core Architecture
1. **Prompt Element Data Model**
2. **Token Budget Controller**
3. **Budget-Aware Prompt Assembly Strategy**
4. **Task-Oriented Generation Configuration**

#### 1. Prompt Element & Priority Control
All prompt segments are encapsulated as `PromptElement`, with four priority levels:
- `CRITICAL`: Must be included (system role, user query)
- `HIGH`: Core content (context, domain rules, plan guidelines)
- `MEDIUM`: Optional content (chat history, user constraints)
- `LOW`: Auxiliary content, truncated first

Each element supports dynamic rendering, content injection and token estimation.

#### 2. Token Budget Management
`TokenBudget` controls overall context window limitation：
- Calculate available prompt tokens by reserving output tokens
- Real-time token usage tracking
- Prevent LLM context overflow
- Built-in `truncate_to_tokens` for long text compression

#### 3. Budget-Aware Assembly Logic
`PromptAssemblyStrategy` follows strict assembly order：
1. Force load all critical & required elements
2. Sort optional elements by priority descending
3. Allocate tokens within remaining budget
4. Auto truncate context/chat history when exceeding limits
5. Record included / excluded / truncated elements for debugging & evaluation

#### 4. Multi-Mode Generation Config
Predefined four task modes with independent hyperparameters：
- `qa`: Low temperature, factual & citation-first
- `plan`: Medium temperature, structured long-form study plan
- `brainstorm`: High temperature, diverse & open generation
- `summarize`: Low temperature, concise key-point output

Each mode contains：
- Temperature / top_p / top_k / num_predict
- Task-specific constraints
- Standard guidance prefix

### Core API
```python
from src.prompt_manager import PromptManager

pm = PromptManager()

# Full assembly with budget & metadata
full_prompt, metadata = pm.assemble_prompt(
    query="Your question",
    context_str=retrieved_context,
    history=chat_history,
    mode="qa"
)

# Simplified prompt only
simple_prompt = pm.assemble_prompt_simple(
    query="7-day study plan",
    context_str=course_context,
    history=chat_history,
    mode="plan",
    user_time="2 hours per day",
    user_goals="Final exam preparation"
)
```

### Module Value
- Unified prompt standard across the whole project
- Effectively avoids context overflow and model OOM
- Supports flexible switching of academic task types
- Traceable prompt quality & token efficiency for evaluation
- Easy to maintain, expand and add new prompt rules

---

## ReAct & LLM-as-a-Judge

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
from judge_engine import JudgeEngine

judge = JudgeEngine(model="gemma3:12b")
result = judge.judge(query="What is RAG?", answer="RAG means ...", context="...")
print(result)
```

---

## Project Structure
```
src/
├─ app.py               # Streamlit web entry (Yan Zibo)
├─ main.py              # CLI demo entry (OU Yuanlin)
├─ rag_engine.py        # Retrieval & embedding cache (Shu Yaming)
├─ document_processor.py# PDF/TXT parsing & chunking (Lin Jing)
├─ chat_logic.py        # RAG pipeline (He Bien)
├─ prompt_manager.py    # Budget-aware prompt engine (OU Yuanlin)
├─ react_engine.py      # ReAct reasoning (Cai Xueying)
├─ react_prompt_manager.py # (Cai Xueying)
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
