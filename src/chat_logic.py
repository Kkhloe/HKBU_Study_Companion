"""
Chat Logic for HKBU Study Companion

"""

import ollama
from typing import List, Dict, Optional
from .rag_engine import RAGEngine
from .prompt_manager import PromptManager, GenerationConfig, TokenBudget

# --- Generation defaults ---
DEFAULT_GENERATION_MODEL = "gemma3:4b"

# Keep last N chat messages (user+assistant pairs count as 2 messages each).
HISTORY_MAX_MESSAGES = 12

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
    are evaluated with conversation context. Uses ``ollama.generate`` with
    ``DEFAULT_GENERATION_MODEL``.
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

    # --- Internal helpers ---

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
    ) -> Dict:
        """
        Process one user turn: detect mode, retrieve context, assemble prompt, generate response.

        Conversation memory: prior turns in ``self.history`` are passed to the prompt manager
        so multi-turn flows (e.g., revising a study plan) work like the notebook demo.
        Set ``update_history=False`` when running side-by-side baselines so one query does not duplicate turns.

        Returns a dict including:
        - response: model text
        - prompt_tokens, completion_tokens, total_tokens: from Ollama eval counts
        - mode, retrieval_type, cited_docs, context_used, budget_info
        - metadata (if return_metadata): prompt assembly and generation config details
        """
        mode = self._detect_mode(query)

        if retrieval_type == "lexical":
            context_str, retrieved_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            context_str, retrieved_chunks = self.rag.neural_search(query, top_k=top_k)

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
            token_budget=token_budget
        )

        gen_config = GenerationConfig(mode=mode)
        gen_params = gen_config.get_generation_params()
        
        response = ollama.generate(
            model=DEFAULT_GENERATION_MODEL,
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
            "cited_docs": list(dict.fromkeys([  # Deduplicate while preserving order
                chunk["metadata"].get("source_path", chunk["metadata"].get("title", "Unknown"))
                for chunk in retrieved_chunks
            ])),
            "context_used": len(retrieved_chunks),
            "budget_info": token_budget.get_budget_info()
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
    ) -> Dict:
        """
        Baseline generation without retrieved documents (no local context), for no-RAG vs RAG comparison.
        Matches the minimal prompt style used in ``notebooks/evaluation_update.ipynb`` / ``evaluation.py``.
        """
        mode = self._detect_mode(query)
        prompt = f"You are HKBU Study Companion.\nUser: {query}\nAssistant: "
        response = ollama.generate(
            model=DEFAULT_GENERATION_MODEL,
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
    ) -> Dict:
        """Run no-RAG and RAG on the same query without appending duplicate history entries."""
        no_rag = self.process_query_no_rag(query, update_history=False)
        rag = self.process_query(
            query, retrieval_type=retrieval_type, top_k=top_k, update_history=False
        )
        return {"no_rag": no_rag, "rag": rag}

    def compare_lexical_vs_neural(self, query: str, top_k: int = 3) -> Dict:
        """Run lexical and neural RAG on the same query without appending duplicate history entries."""
        lexical = self.process_query(
            query, retrieval_type="lexical", top_k=top_k, update_history=False
        )
        neural = self.process_query(
            query, retrieval_type="neural", top_k=top_k, update_history=False
        )
        return {"lexical": lexical, "neural": neural}

    # --- Advanced / utilities ---

    def process_query_advanced(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        max_prompt_tokens: int = 3200,
        max_output_tokens: int = 800
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
        )

    def clear_history(self):
        """Clear multi-turn conversation history (e.g., before a new demo or isolated task)."""
        self.history.clear()

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
