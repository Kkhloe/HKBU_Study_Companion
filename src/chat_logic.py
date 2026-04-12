"""
Chat Logic for HKBU Study Companion

"""

import re
import ollama
from typing import List, Dict, Optional

# Notebook: `from src.chat_logic import ChatLogic` (package). App/streamlit: `src` on sys.path (flat).
try:
    from .rag_engine import RAGEngine
    from .prompt_manager import PromptManager, GenerationConfig, TokenBudget
except ImportError:
    from rag_engine import RAGEngine
    from prompt_manager import PromptManager, GenerationConfig, TokenBudget

# --- Generation defaults ---
DEFAULT_GENERATION_MODEL = "gemma3:4b"

# Keep last N chat messages (user+assistant pairs count as 2 messages each).
HISTORY_MAX_MESSAGES = 12

# HKBU-style course codes appearing in user text or syllabus citations (e.g. COMP7300, COMP7125)
_COURSE_CODE_RE = re.compile(r"\b(COMP\d{4}[A-Z]?)\b", re.IGNORECASE)

# Follow-ups that should reuse the last retrieval (reformat prior answer, not a new topic search)
_REUSE_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"rewrite|re-?write|rephrase|"
    r"one\s+short\s+paragraph|in\s+one\s+paragraph|"
    r"shorter|more\s+concise|briefly|"
    r"same\s+answer\s+but|format\s+as\s+a\s+paragraph|"
    r"translate\s+(the\s+)?(above|previous|last)|"
    r"turn\s+(the\s+)?(above|last)\s+into"
    r")\b"
)

# --- Reference text (docs/tests; not sent to the model unless you import and use it) ---
STUDY_PLAN_ASSISTANT_OUTPUT_GUIDELINES = """
Expected assistant structure for study-plan responses (see PromptManager plan templates):
- Per day: which course(s), hours, concrete knowledge points from syllabus/context
- Per knowledge point: target mastery (e.g., Familiar / Proficient / Exam-ready or Bloom level)
- Assumptions, risk flags, citations and cited_docs as before
"""


class ChatLogic:
    """
    Conversation orchestration: mode detection, RAG retrieval, prompt assembly, and generation.

    ``history`` stores prior user/assistant turns so follow-up queries (e.g., updating a plan)
    are evaluated with conversation context. Uses ``ollama.generate``; pass ``model=`` per call or
    ``DEFAULT_GENERATION_MODEL`` by default.
    """

    def __init__(self, rag_engine: RAGEngine, prompt_manager: PromptManager):
        self.rag = rag_engine
        self.prompt_manager = prompt_manager
        self.history: List[Dict] = []
        self.token_usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "queries_processed": 0
        }
        # Multi-turn RAG: last successful retrieval (for rewrite-style follow-ups)
        self._last_context_str: Optional[str] = None
        self._last_retrieved_chunks: Optional[List[Dict]] = None

    # --- Internal helpers ---

    def _history_blob(self) -> str:
        return " ".join(m.get("content", "") for m in self.history)

    @staticmethod
    def _extract_course_codes(*texts: str) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for t in texts:
            for m in _COURSE_CODE_RE.findall(t or ""):
                u = m.upper()
                if u not in seen:
                    seen.add(u)
                    ordered.append(u)
        return ordered[:6]

    def _build_retrieval_query(self, query: str) -> str:
        """
        Expand the user query with course codes mentioned in the current turn or prior history
        so follow-ups like "narrow that to assessment" still retrieve the same course.
        """
        codes = self._extract_course_codes(query, self._history_blob())
        if not codes:
            return query.strip()
        return f"{' '.join(codes)} {query}".strip()

    def _should_reuse_last_retrieval(self, query: str) -> bool:
        if not self._last_context_str or not self.history:
            return False
        if len(self.history) < 2:
            return False
        return _REUSE_CONTEXT_RE.search(query.strip()) is not None

    def _append_history_turn(self, query: str, assistant_text: str) -> None:
        """Append one user/assistant exchange and trim to HISTORY_MAX_MESSAGES."""
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": assistant_text})
        if len(self.history) > HISTORY_MAX_MESSAGES:
            self.history = self.history[-HISTORY_MAX_MESSAGES:]

    def _accumulate_token_stats(self, result: Dict) -> None:
        self.token_usage_stats["total_prompt_tokens"] += result["prompt_tokens"]
        self.token_usage_stats["total_completion_tokens"] += result["completion_tokens"]
        self.token_usage_stats["total_tokens"] += result["total_tokens"]
        self.token_usage_stats["queries_processed"] += 1

    # --- Mode detection ---

    def _detect_mode(self, query: str) -> str:
        """Auto-detect task mode: qa, plan, brainstorm, or summarize (keyword-based)."""
        query_lower = query.lower()
        
        plan_keywords = [
            "study plan", "plan", "schedule", "time table", 
            "weekly", "2-week", "3-week", "organize"
        ]
        if any(k in query_lower for k in plan_keywords):
            return "plan"
        
        summarize_keywords = ["summarize", "summary", "sum up", "condense", "brief", "overview"]
        if any(k in query_lower for k in summarize_keywords):
            return "summarize"
        
        brainstorm_keywords = ["ideas", "suggestions", "approaches", "options", "alternatives"]
        if any(k in query_lower for k in brainstorm_keywords):
            return "brainstorm"
        
        return "qa"

    # --- Main path: retrieve → assemble prompt → generate ---

    def process_query(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        return_metadata: bool = False,
        token_budget: Optional[TokenBudget] = None,
        update_history: bool = True,
        use_react_override: bool = False,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        """
        Process one user turn: detect mode, retrieve context, assemble prompt, generate response.

        ``model``: Ollama model name for generation (defaults to ``DEFAULT_GENERATION_MODEL``).

        Conversation memory: prior turns in ``self.history`` are passed to the prompt manager
        so multi-turn flows (e.g., revising a study plan) work like the notebook demo.
        Set ``update_history=False`` when running side-by-side baselines so one query does not duplicate turns.

        Structured constraints (optional) are forwarded to ``PromptManager.assemble_prompt`` so study-plan
        prompts can include a USER CONSTRAINTS block (time, goals, workload).

        Returns a dict including:
        - response: model text
        - prompt_tokens, completion_tokens, total_tokens: from Ollama eval counts
        - mode, retrieval_type, cited_docs, context_used, budget_info, model
        - metadata (if return_metadata): prompt assembly and generation config details
        """
        mode = self._detect_mode(query)
        resolved_model = model or DEFAULT_GENERATION_MODEL

        # 判断是否启用 ReAct
        if hasattr(self, "react_engine") and getattr(self, "react_enabled", False):
            # 使用 ReAct 推理（与主链路使用同一生成模型）
            self.react_engine.model = resolved_model
            rq = self._build_retrieval_query(query)
            context_str, retrieved_chunks = self.rag.neural_search(rq, top_k=top_k)
            react_result = self.react_engine.reason(query, context=context_str)
            result = {
                "response": react_result["final_answer"],
                "reasoning_enabled": True,
                "reasoning_steps": react_result["reasoning_steps"],
                "reasoning_trace": self.react_engine.get_reasoning_trace(),
                "prompt_tokens": react_result.get("prompt_tokens", 0),
                "completion_tokens": react_result.get("completion_tokens", 0),
                "total_tokens": react_result.get("total_tokens", 0),
                "mode": mode,
                "retrieval_type": retrieval_type,
                "cited_docs": list(dict.fromkeys([
                    chunk["metadata"].get("source_path", chunk["metadata"].get("title", "Unknown"))
                    for chunk in retrieved_chunks
                ])),
                "context_used": len(retrieved_chunks),
                "budget_info": {"note": "ReAct reasoning mode"},
                "model": resolved_model,
            }
            if update_history:
                self._append_history_turn(query, react_result["final_answer"])
            self._accumulate_token_stats(result)
            return result

        # 普通模式：检索查询带课程码（多轮）；改写类跟进可复用上一轮 context
        context_reused = False
        if self._should_reuse_last_retrieval(query):
            context_str = self._last_context_str or ""
            retrieved_chunks = list(self._last_retrieved_chunks or [])
            retrieval_query = "(reused previous turn context)"
            context_reused = True
        else:
            retrieval_query = self._build_retrieval_query(query)
            if retrieval_type == "lexical":
                context_str, retrieved_chunks = self.rag.lexical_search(retrieval_query, top_k=top_k)
            else:
                context_str, retrieved_chunks = self.rag.neural_search(retrieval_query, top_k=top_k)
            if update_history:
                self._last_context_str = context_str
                self._last_retrieved_chunks = retrieved_chunks

        if token_budget is None:
            token_budget = TokenBudget(
                total_budget=4096,
                reserved_for_output=800 if mode == "qa" else 1000
            )
        prompt, prompt_metadata = self.prompt_manager.assemble_prompt(
            query=query,
            context_str=context_str,
            history=self.history,
            mode=mode,
            token_budget=token_budget,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload,
        )
        gen_config = GenerationConfig(mode=mode)
        gen_params = gen_config.get_generation_params()
        response = ollama.generate(
            model=resolved_model,
            prompt=prompt,
            options=gen_params
        )
        response_text = response["response"].strip()
        result = {
            "response": response_text,
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": (
                response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            ),
            "mode": mode,
            "retrieval_type": retrieval_type,
            "cited_docs": list(dict.fromkeys([
                chunk["metadata"].get("source_path", chunk["metadata"].get("title", "Unknown"))
                for chunk in retrieved_chunks
            ])),
            "context_used": len(retrieved_chunks),
            "budget_info": token_budget.get_budget_info(),
            "model": resolved_model,
            "retrieval_query": retrieval_query,
            "context_reused": context_reused,
        }
        if user_time or user_goals or user_workload:
            result["user_constraints"] = {
                "user_time": user_time,
                "user_goals": user_goals,
                "user_workload": user_workload,
            }
        if return_metadata:
            result["metadata"] = {
                "prompt_metadata": prompt_metadata,
                "generation_config": gen_config.to_dict(),
                "full_prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt
            }
        if update_history:
            self._append_history_turn(query, response_text)
        self._accumulate_token_stats(result)
        return result

    # --- Baselines (no-RAG / retrieval comparisons) ---

    def process_query_no_rag(
        self,
        query: str,
        update_history: bool = True,
        model: Optional[str] = None,
    ) -> Dict:
        """
        Baseline generation without retrieved documents (no local context), for no-RAG vs RAG comparison.
        Matches the minimal prompt style used in ``notebooks/evaluation_update.ipynb`` / ``evaluation.py``.
        """
        mode = self._detect_mode(query)
        resolved_model = model or DEFAULT_GENERATION_MODEL
        prompt = f"You are HKBU Study Companion.\nUser: {query}\nAssistant: "
        response = ollama.generate(
            model=resolved_model,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 600},
        )
        response_text = response["response"].strip()
        result = {
            "response": response_text,
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": (
                response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            ),
            "mode": mode,
            "retrieval_type": "none",
            "cited_docs": [],
            "context_used": 0,
            "budget_info": {"note": "no-RAG baseline; no TokenBudget assembly"},
            "model": resolved_model,
        }

        if update_history:
            self._append_history_turn(query, response_text)

        self._accumulate_token_stats(result)

        return result

    def compare_no_rag_vs_rag(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        """Run no-RAG and RAG on the same query without appending duplicate history entries."""
        no_rag = self.process_query_no_rag(query, update_history=False, model=model)
        rag = self.process_query(
            query,
            retrieval_type=retrieval_type,
            top_k=top_k,
            update_history=False,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload,
            model=model,
        )
        return {"no_rag": no_rag, "rag": rag}

    def compare_lexical_vs_neural(
        self,
        query: str,
        top_k: int = 3,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        """Run lexical and neural RAG on the same query without appending duplicate history entries."""
        kw = dict(
            top_k=top_k,
            update_history=False,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload,
            model=model,
        )
        lexical = self.process_query(query, retrieval_type="lexical", **kw)
        neural = self.process_query(query, retrieval_type="neural", **kw)
        return {"lexical": lexical, "neural": neural}

    # --- Advanced / utilities ---
    def enable_react(self, react_engine):
        """启用 ReAct 推理引擎"""
        self.react_engine = react_engine
        self.react_enabled = True

    def disable_react(self):
        """关闭 ReAct"""
        self.react_enabled = False

    def process_query_advanced(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        max_prompt_tokens: int = 3200,
        max_output_tokens: int = 800,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        """Advanced query processing with custom token budgets; calls process_query with return_metadata=True."""
        token_budget = TokenBudget(
            total_budget=max_prompt_tokens + max_output_tokens,
            reserved_for_output=max_output_tokens
        )
        
        return self.process_query(
            query=query,
            retrieval_type=retrieval_type,
            top_k=top_k,
            return_metadata=True,
            token_budget=token_budget,
            update_history=True,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload,
            model=model,
        )

    def clear_history(self):
        """Clear multi-turn conversation history (e.g., before a new demo or isolated task)."""
        self.history.clear()
        self._last_context_str = None
        self._last_retrieved_chunks = None

    def get_token_stats(self) -> Dict:
        """Get token usage statistics"""
        return self.token_usage_stats

    def reset_token_stats(self):
        """Reset token statistics"""
        self.token_usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "queries_processed": 0
        }

    def get_system_info(self) -> Dict:
        """Get system information and statistics"""
        return {
            "rag_engine": type(self.rag).__name__,
            "prompt_manager": type(self.prompt_manager).__name__,
            "history_length": len(self.history),
            "available_modes": ["qa", "plan", "brainstorm", "summarize"],
            "token_stats": self.token_usage_stats
        }
