"""
Chat Logic for HKBU Study Companion

"""

import ollama
from typing import List, Dict, Tuple, Optional, Callable
from rag_engine import RAGEngine
from prompt_manager import PromptManager, GenerationConfig, TokenBudget
from LLM_judge_engine import JudgeEngine



class ChatLogic:
    def __init__(self, rag_engine: RAGEngine, prompt_manager: PromptManager):
        self.rag = rag_engine
        self.prompt_manager = prompt_manager
        self.history: List[Dict] = [] # History of Multi-Turn Dialogue
        self.token_usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "queries_processed": 0
        }  
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
        token_budget: Optional[TokenBudget] = None,
        use_react_override: Optional[bool] = None
    ) -> Dict:
        """Process user query: detect mode, retrieve context, assemble prompt, generate response
        """

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
        

        mode = self._detect_mode(query)

        if retrieval_type == "lexical":
            context_str, retrieved_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            context_str, retrieved_chunks = self.rag.neural_search(query, top_k=top_k)
        
        # Post-hook
        if self.on_after_retrieval:
            self.on_after_retrieval(query, retrieved_chunks)

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
        
        # 3. Generation Control + pre-hook
        if self.on_before_generation:
            self.on_before_generation(query, context_str)
        
        temperature = 0.0 if mode == "qa" else 0.7
        
        
        response = ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            options=gen_params
        )

        # Post-hook
        if self.on_after_generation:
            self.on_after_generation(response["response"].strip())

        response_text = response["response"].strip()
        
        # 4. Results Processing
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
            "reasoning_enabled": False,
            "budget_info": token_budget.get_budget_info()
        }

        if return_metadata:
            result["metadata"] = {
                "prompt_metadata": prompt_metadata,
                "generation_config": gen_config.to_dict(),
                "full_prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt
            }

        # 5. Update conversation history (Conversation Management)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response_text})

        if len(self.history) > 12:
            self.history = self.history[-12:]

        self.token_usage_stats["total_prompt_tokens"] += result["prompt_tokens"]
        self.token_usage_stats["total_completion_tokens"] += result["completion_tokens"]
        self.token_usage_stats["total_tokens"] += result["total_tokens"]
        self.token_usage_stats["queries_processed"] += 1

        return result

    def _process_query_with_react(
        self,
        query: str,
        retrieval_type: str,
        top_k: int,
        token_budget=None  # 把 token budget 加进来
    ) -> Dict:
        """
        ReAct reasoning pipeline with advanced token budget control
        整合：ReAct 分步推理 + 高级 Token 预算管理
        """
        # 获取初始上下文
        if retrieval_type == "lexical":
            init_context, init_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            init_context, init_chunks = self.rag.neural_search(query, top_k=top_k)
        
        # 执行 ReAct 推理
        react_result = self.react_engine.reason(
            query=query,
            context=init_context,
            retrieval_callback=lambda q: self.rag.neural_search(q, top_k=top_k)
        )
        
        # 构建返回结果（同时保留 token 信息 + ReAct 信息）
        result = {
            "response": react_result["final_answer"],
            "prompt_tokens": 0,
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
            "reasoning_steps": react_result["reasoning_steps"],
            "reasoning_trace": self.react_engine.get_reasoning_trace(),
            "budget_info": token_budget.get_budget_info() if token_budget else None  # 整合预算
        }
        
        # 更新对话历史
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result["response"]})
        
        if len(self.history) > 12:
            self.history = self.history[-12:]
        
        return result


    def process_query_advanced(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        max_prompt_tokens: int = 3200,
        max_output_tokens: int = 800
    ) -> Dict:
        """高级查询入口：自动使用 ReAct + Token 预算"""
        token_budget = TokenBudget(
            total_budget=max_prompt_tokens + max_output_tokens,
            reserved_for_output=max_output_tokens
        )

        # 直接调用整合后的 ReAct 函数
        return self._process_query_with_react(
            query=query,
            retrieval_type=retrieval_type,
            top_k=top_k,
            token_budget=token_budget  # 传入预算
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