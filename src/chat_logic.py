"""
Chat Logic for HKBU Study Companion

"""

import ollama
from typing import List, Dict, Tuple, Optional, Callable
from rag_engine import RAGEngine
from prompt_manager import PromptManager
from LLM_judge_engine import JudgeEngine

class ChatLogic:
    def __init__(self, rag_engine: RAGEngine, prompt_manager: PromptManager):
        self.rag = rag_engine
        self.prompt_manager = prompt_manager
        self.history: List[Dict] = []         # History of Multi-Turn Dialogue
        self.judge_engine = JudgeEngine()
        
        # ==================== Hook System (Plugin-style ReAct Interface) ====================
        """
        Hook points supported (plugin-style ReAct integration):
        - on_before_retrieval: before retrieval hook
        - on_after_retrieval: after retrieval hook
        - on_before_generation: before generation hook
        - on_after_generation: after generation hook

        """
        self.use_react = False                 # Whether to enable ReAct reasoning
        self.react_engine = None               # ReAct engine instance
        
        # Hook callback functions
        self.on_before_retrieval: Optional[Callable[[str], None]] = None
        self.on_after_retrieval: Optional[Callable[[str, List[Dict]], None]] = None
        self.on_before_generation: Optional[Callable[[str, str], None]] = None
        self.on_after_generation: Optional[Callable[[str], None]] = None

    def judge_response(self, query, answer, context=""):
        return self.judge_engine.judge(query, answer, context)
    
    def enable_react(self, react_engine: 'ReActEngine'):
        """Enable ReAct reasoning engine"""
        from react_engine import ReActEngine
        self.react_engine = react_engine
        self.use_react = True
        # Register search callback for the ReAct engine
        self.react_engine.set_search_callback(self._react_search_callback)
    
    def disable_react(self):
        """Disable ReAct reasoning engine"""
        self.use_react = False
        self.react_engine = None
    
    def _react_search_callback(self, query: str) -> Tuple[str, List[Dict]]:
        """Search callback used by ReAct engine"""
        context_str, chunks = self.rag.neural_search(query, top_k=3)
        return context_str, chunks
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register custom hook"""
        valid_hooks = [
            'on_before_retrieval',
            'on_after_retrieval',
            'on_before_generation',
            'on_after_generation'
        ]
        if hook_name in valid_hooks:
            setattr(self, hook_name, callback)
        else:
            raise ValueError(f"Unknown hook: {hook_name}. Valid hooks: {valid_hooks}")

    def _detect_mode(self, query: str) -> str:
        """Automatically identify whether it is a learning plan request"""
        plan_keywords = ["study plan", "Study plan", "scheduling", "schedule", "time table", "weekly plan", "2-week", "3-week"]
        query_lower = query.lower()
        return "plan" if any(k in query_lower for k in plan_keywords) else "qa"

    def process_query(
        self,
        query: str,
        retrieval_type: str = "neural",   # "neural" or "lexical"
        top_k: int = 3,
        use_react_override: Optional[bool] = None
    ) -> Dict:
        """
        Complete Processing Flow (Feedforward Pass + Application Loop):
        1. Retrieval (lexical / neural) + hook support
        2. Prompt assembly
        3. Generation control (temperature + num_predict) + optional ReAct reasoning
        4. Update dialogue history
        5. Return results with token and citation
        
        Parameters:
        - use_react_override: temporarily override global ReAct setting
        """
        
        # Determine whether to use ReAct
        use_react = use_react_override if use_react_override is not None else self.use_react
        
        # ============ If ReAct is enabled, use reasoning pipeline ============
        if use_react and self.react_engine:
            return self._process_query_with_react(query, retrieval_type, top_k)
        
        # ============ Otherwise, use original standard flow (fully compatible) ============
        mode = self._detect_mode(query)

        # 1. Retrieval + pre-hook
        if self.on_before_retrieval:
            self.on_before_retrieval(query)
        
        if retrieval_type == "lexical":
            context_str, retrieved_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            context_str, retrieved_chunks = self.rag.neural_search(query, top_k=top_k)
        
        # Post-hook
        if self.on_after_retrieval:
            self.on_after_retrieval(query, retrieved_chunks)

        # 2. Prompt Assembly
        prompt = self.prompt_manager.assemble_prompt(
            query=query,
            context_str=context_str,
            history=self.history,
            mode=mode
        )

        # 3. Generation Control + pre-hook
        if self.on_before_generation:
            self.on_before_generation(query, context_str)
        
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

        # Post-hook
        if self.on_after_generation:
            self.on_after_generation(response["response"].strip())

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
            "context_used": len(retrieved_chunks),
            "reasoning_enabled": False
        }

        # 5. Update conversation history (Conversation Management)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result["response"]})

        # Optional: Automatically truncate when the history is too long (to prevent exceeding the context).
        if len(self.history) > 12:  # Approximately 6 rounds of dialogue
            self.history = self.history[-12:]

        return result

    def _process_query_with_react(self, query: str, retrieval_type: str, top_k: int) -> Dict:
        """
        ReAct reasoning pipeline: use step-by-step reasoning instead of one-shot generation
        """
        # Get initial context
        if retrieval_type == "lexical":
            init_context, init_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            init_context, init_chunks = self.rag.neural_search(query, top_k=top_k)
        
        # Execute ReAct reasoning
        react_result = self.react_engine.reason(
            query=query,
            context=init_context,
            retrieval_callback=lambda q: self.rag.neural_search(q, top_k=top_k)
        )
        
        # Build response (maintain compatible interface with standard flow)
        result = {
            "response": react_result["final_answer"],
            "prompt_tokens": 0,  # Standard tokens not used in ReAct
            "completion_tokens": 0,
            "total_tokens": 0,
            "mode": self._detect_mode(query),
            "retrieval_type": retrieval_type,
            "cited_docs": [
                chunk["metadata"].get("source_path", chunk["metadata"].get("title", "Unknown"))
                for chunk in init_chunks
            ],
            "context_used": len(init_chunks),
            "reasoning_enabled": True,
            "reasoning_steps": react_result["reasoning_steps"],  # Extra reasoning process
            "reasoning_trace": self.react_engine.get_reasoning_trace()
        }
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result["response"]})
        
        if len(self.history) > 12:
            self.history = self.history[-12:]
        
        return result

    def clear_history(self):
        """Clear conversation history (for resetting)"""
        self.history.clear()