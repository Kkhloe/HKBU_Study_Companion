"""
Chat Logic for HKBU Study Companion
Handles conversation flow, mode detection, RAG calls, and token management
"""

import ollama
from typing import List, Dict, Optional
from .rag_engine import RAGEngine
from .prompt_manager import PromptManager, GenerationConfig, TokenBudget


class ChatLogic:
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

    def _detect_mode(self, query: str) -> str:
        """Auto-detect task mode: qa, plan, brainstorm, or summarize"""
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

    def process_query(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        return_metadata: bool = False,
        token_budget: Optional[TokenBudget] = None
    ) -> Dict:
        """Process user query: detect mode, retrieve context, assemble prompt, generate response"""
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
            model="gemma3:4b",
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

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response_text})

        if len(self.history) > 12:
            self.history = self.history[-12:]

        self.token_usage_stats["total_prompt_tokens"] += result["prompt_tokens"]
        self.token_usage_stats["total_completion_tokens"] += result["completion_tokens"]
        self.token_usage_stats["total_tokens"] += result["total_tokens"]
        self.token_usage_stats["queries_processed"] += 1

        return result

    def process_query_advanced(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        max_prompt_tokens: int = 3200,
        max_output_tokens: int = 800
    ) -> Dict:
        """Advanced query processing with custom token budgets"""
        token_budget = TokenBudget(
            total_budget=max_prompt_tokens + max_output_tokens,
            reserved_for_output=max_output_tokens
        )
        
        return self.process_query(
            query=query,
            retrieval_type=retrieval_type,
            top_k=top_k,
            return_metadata=True,
            token_budget=token_budget
        )

    def clear_history(self):
        """Clear conversation history"""
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