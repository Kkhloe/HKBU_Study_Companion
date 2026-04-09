"""
Chat Logic for HKBU Study Companion
负责多轮对话、模式识别、RAG 调用、生成控制和 Token 统计
完全适配优化后的 rag_engine.py
"""

import ollama
from typing import List, Dict, Tuple
from .rag_engine import RAGEngine
from .prompt_manager import PromptManager

class ChatLogic:
    def __init__(self, rag_engine: RAGEngine, prompt_manager: PromptManager):
        self.rag = rag_engine
        self.prompt_manager = prompt_manager
        self.history: List[Dict] = []         # History of Multi-Turn Dialogue

    def _detect_mode(self, query: str) -> str:
        """Automatically identify whether it is a learning plan request"""
        plan_keywords = ["study plan", "Study plan", "scheduling", "schedule", "time table", "weekly plan", "2-week", "3-week"]
        query_lower = query.lower()
        return "plan" if any(k in query_lower for k in plan_keywords) else "qa"

    def process_query(
        self,
        query: str,
        retrieval_type: str = "neural",   # "neural" or "lexical"
        top_k: int = 3
    ) -> Dict:
        """
        Complete Processing Flow (Feedforward Pass + Application Loop):
        1. Retrieval (lexical / neural)
        2. Cue word assembly
        3. Generation control (temperature + num_predict)
        4. Update dialogue history
        5. Return results with token and citation
        """
        mode = self._detect_mode(query)

        # 1. Retrieval
        if retrieval_type == "lexical":
            context_str, retrieved_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            context_str, retrieved_chunks = self.rag.neural_search(query, top_k=top_k)

        # 2. Prompt Assembly
        prompt = self.prompt_manager.assemble_prompt(
            query=query,
            context_str=context_str,
            history=self.history,
            mode=mode
        )

        # 3. Generation Control
        temperature = 0.0 if mode == "qa" else 0.7
        response = ollama.generate(
            model="gemma3:4b",          # Can be exchanged qwen3:9b
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": 800,    # Control output length
                "top_p": 0.9,
                "top_k": 40,
            }
        )

        # 4. Results Processing
        result = {
            "response": response["response"].strip(),
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": (
                response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            ),
            "mode": mode,
            "retrieval_type": retrieval_type,
            "cited_docs": [
                chunk["metadata"].get("source_path", chunk["metadata"].get("title", "Unknown"))
                for chunk in retrieved_chunks
            ],
            "context_used": len(retrieved_chunks)
        }

        # 5. Update conversation history（Conversation Management）
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result["response"]})

        # Optional: Automatically truncate when the history is too long (to prevent exceeding the context).
        if len(self.history) > 12:  # Approximately 6 rounds of dialogue
            self.history = self.history[-12:]

        return result

    def clear_history(self):
        """Clear conversation history (for resetting)"""
        self.history.clear()