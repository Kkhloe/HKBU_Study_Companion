"""
Chat Logic for HKBU Study Companion
负责多轮对话、模式识别、RAG 调用、生成控制和 Token 统计
完全适配优化后的 rag_engine.py 和增强的 prompt_manager.py
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
        """
        自动检测任务模式
        
        Modes:
        - "qa": 普通问答
        - "plan": 学习计划
        - "brainstorm": 头脑风暴
        - "summarize": 总结
        """
        query_lower = query.lower()
        
        # 学习计划检测
        plan_keywords = [
            "study plan", "plan", "schedule", "schedule", "time table", 
            "weekly", "2-week", "3-week", "organize", "organize", "plan out"
        ]
        if any(k in query_lower for k in plan_keywords):
            return "plan"
        
        # 总结检测
        summarize_keywords = ["summarize", "summary", "sum up", "condense", "brief", "overview"]
        if any(k in query_lower for k in summarize_keywords):
            return "summarize"
        
        # 头脑风暴检测
        brainstorm_keywords = ["ideas", "suggestions", "approaches", "options", "alternatives", "think of"]
        if any(k in query_lower for k in brainstorm_keywords):
            return "brainstorm"
        
        # 默认为问答
        return "qa"

    def process_query(
        self,
        query: str,
        retrieval_type: str = "neural",
        top_k: int = 3,
        return_metadata: bool = False,
        token_budget: Optional[TokenBudget] = None
    ) -> Dict:
        """
        完整的查询处理流程
        
        Flow:
        1. 模式检测
        2. 检索（词法/神经）
        3. Token预算管理
        4. 提示词组装（使用数据化定义）
        5. 生成控制（使用GenerationConfig）
        6. 更新对话历史
        7. 统计Token和引用
        
        Args:
            query: 用户查询
            retrieval_type: 检索类型（"neural"或"lexical"）
            top_k: 检索的上下文数量
            return_metadata: 是否返回详细的元数据
            token_budget: 自定义token预算（可选）
        
        Returns:
            结果字典，包含response、tokens、mode等
        """
        # 1. 模式检测
        mode = self._detect_mode(query)

        # 2. 检索
        if retrieval_type == "lexical":
            context_str, retrieved_chunks = self.rag.lexical_search(query, top_k=top_k)
        else:
            context_str, retrieved_chunks = self.rag.neural_search(query, top_k=top_k)

        # 3. Token预算管理
        if token_budget is None:
            token_budget = TokenBudget(
                total_budget=4096,
                reserved_for_output=800 if mode == "qa" else 1000
            )
        
        # 4. 提示词组装（预算感知）
        prompt, prompt_metadata = self.prompt_manager.assemble_prompt(
            query=query,
            context_str=context_str,
            history=self.history,
            mode=mode,
            token_budget=token_budget
        )

        # 5. 生成控制
        gen_config = GenerationConfig(mode=mode)
        gen_params = gen_config.get_generation_params()
        
        response = ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            options=gen_params
        )

        # 6. 结果处理
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

        # 可选：返回详细元数据
        if return_metadata:
            result["metadata"] = {
                "prompt_metadata": prompt_metadata,
                "generation_config": gen_config.to_dict(),
                "full_prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt
            }

        # 7. 更新对话历史
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response_text})

        # 历史管理：太长时自动裁剪
        if len(self.history) > 12:  # 约6轮对话
            self.history = self.history[-12:]

        # 8. 更新统计
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
        """
        高级查询处理：完整控制所有参数
        """
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
        """清空对话历史"""
        self.history.clear()

    def get_token_stats(self) -> Dict:
        """获取Token使用统计"""
        return self.token_usage_stats

    def reset_token_stats(self):
        """重置Token统计"""
        self.token_usage_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "queries_processed": 0
        }

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            "rag_engine": type(self.rag).__name__,
            "prompt_manager": type(self.prompt_manager).__name__,
            "history_length": len(self.history),
            "available_modes": ["qa", "plan", "brainstorm", "summarize"],
            "token_stats": self.token_usage_stats
        }