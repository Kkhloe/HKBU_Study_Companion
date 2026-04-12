"""
Prompt Manager for HKBU Study Companion
负责构建高质量提示词，完全符合课程 Topic 6 最佳实践
包含：提示词元素数据化定义、预算感知的组装策略、生成控制与引导
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json


# ============================================================
# Part 1: Prompt Element Data Model (数据化定义)
# ============================================================

class ElementPriority(Enum):
    """元素优先级：用于预算约束时的裁剪"""
    CRITICAL = 4      # 不能删除（系统提示、用户查询）
    HIGH = 3           # 尽量保留（核心规则、上下文）
    MEDIUM = 2         # 可选保留（对话历史）
    LOW = 1             # 优先删除（辅助信息）


@dataclass
class PromptElement:
    """
    提示词元素的数据化定义
    
    Attributes:
        name: 元素名称
        template: 元素模板（可包含{占位符}）
        content: 实际内容或生成该元素的函数
        token_estimate: 预估token数（用于预算管理）
        priority: 优先级（用于token不足时裁剪）
        required: 是否必须包含
        parameters: 元素参数字典
    """
    name: str
    template: str
    content: str = ""
    token_estimate: int = 0
    priority: ElementPriority = ElementPriority.MEDIUM
    required: bool = False
    parameters: Dict = field(default_factory=dict)
    
    def render(self, **kwargs) -> Tuple[str, int]:
        """
        渲染元素为文本
        
        Returns:
            (rendered_text, estimated_tokens)
        """
        try:
            rendered = self.template.format(content=self.content, **kwargs, **self.parameters)
        except (KeyError, IndexError):
            rendered = self.template
        
        # 更新token估计
        estimated_tokens = len(rendered.split()) + 10  # 粗略估计
        return rendered, estimated_tokens


@dataclass
class TokenBudget:
    """
    Token预算管理器
    
    Attributes:
        total_budget: 总token预算
        used_tokens: 已使用tokens
        reserved_for_output: 为输出保留的tokens
    """
    total_budget: int = 4096
    used_tokens: int = 0
    reserved_for_output: int = 800
    
    @property
    def available_for_prompt(self) -> int:
        """可用于提示词的tokens"""
        return self.total_budget - self.reserved_for_output - self.used_tokens
    
    @property
    def remaining_budget(self) -> int:
        """剩余预算"""
        return max(0, self.available_for_prompt)
    
    def allocate(self, tokens: int) -> bool:
        """分配tokens，返回是否成功"""
        if self.used_tokens + tokens <= self.available_for_prompt:
            self.used_tokens += tokens
            return True
        return False
    
    def reset(self):
        """重置预算"""
        self.used_tokens = 0
    
    def get_budget_info(self) -> Dict:
        """获取预算信息"""
        return {
            "total": self.total_budget,
            "used": self.used_tokens,
            "available": self.available_for_prompt,
            "remaining": self.remaining_budget,
            "reserved_for_output": self.reserved_for_output
        }


# ============================================================
# Part 2: Prompt Assembly Strategy (预算感知的组装策略)
# ============================================================

class PromptAssemblyStrategy:
    """
    预算感知的提示词组装策略
    实现灵活的元素选择和优先级裁剪
    """
    
    def __init__(self, max_context_tokens: int = 1500):
        self.max_context_tokens = max_context_tokens
        self.elements: List[PromptElement] = []
    
    def add_element(self, element: PromptElement) -> None:
        """添加提示词元素"""
        self.elements.append(element)
    
    def prioritize_elements(self) -> List[PromptElement]:
        """按优先级排序元素"""
        return sorted(self.elements, key=lambda e: e.priority.value, reverse=True)
    
    def assemble_with_budget(self, budget: TokenBudget) -> Tuple[str, Dict]:
        """
        预算感知的组装策略
        
        Strategy:
        1. 先加入CRITICAL优先级元素（必须）
        2. 再按优先级加入其他元素
        3. 如果超预算，按反向优先级裁剪
        
        Returns:
            (assembled_prompt, metadata)
        """
        assembled_parts = []
        element_metadata = {
            "included_elements": [],
            "excluded_elements": [],
            "total_tokens_used": 0
        }
        
        # 1. 先加入必须元素
        critical_elements = [e for e in self.elements if e.required or e.priority == ElementPriority.CRITICAL]
        for elem in critical_elements:
            rendered, tokens = elem.render()
            if budget.allocate(tokens):
                assembled_parts.append(rendered)
                element_metadata["included_elements"].append({
                    "name": elem.name,
                    "tokens": tokens,
                    "priority": elem.priority.name
                })
            else:
                # 强制加入，即使超预算（CRITICAL必须包含）
                if elem.required:
                    assembled_parts.append(rendered)
                    element_metadata["included_elements"].append({
                        "name": elem.name,
                        "tokens": tokens,
                        "priority": elem.priority.name,
                        "forced": True
                    })
        
        # 2. 按优先级加入其他元素
        optional_elements = [e for e in self.elements 
                            if not e.required and e.priority != ElementPriority.CRITICAL]
        optional_elements.sort(key=lambda e: e.priority.value, reverse=True)
        
        for elem in optional_elements:
            rendered, tokens = elem.render()
            if budget.allocate(tokens):
                assembled_parts.append(rendered)
                element_metadata["included_elements"].append({
                    "name": elem.name,
                    "tokens": tokens,
                    "priority": elem.priority.name
                })
            else:
                element_metadata["excluded_elements"].append({
                    "name": elem.name,
                    "tokens": tokens,
                    "priority": elem.priority.name,
                    "reason": "budget_exceeded"
                })
        
        element_metadata["total_tokens_used"] = budget.used_tokens
        assembled_prompt = "\n\n".join(assembled_parts)
        
        return assembled_prompt, element_metadata


# ============================================================
# Part 3: Generation Control & Guidance (生成控制与引导)
# ============================================================

class GenerationConfig:
    """
    生成配置：定义不同任务类型的生成参数
    """
    
    def __init__(self, mode: str = "qa"):
        self.mode = mode
        self.params = self._get_default_params(mode)
    
    def _get_default_params(self, mode: str) -> Dict:
        """根据任务类型返回默认生成参数"""
        
        params_map = {
            "qa": {
                "temperature": 0.0,         # 低温度：精确答题
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 500,         # 适中输出长度
                "guidance_prefix": "Based on the provided context, my answer is:",
                "constraints": [
                    "Must cite the source",
                    "Only use provided context",
                    "Be concise and accurate"
                ]
            },
            "plan": {
                "temperature": 0.7,         # 中温度：创意规划
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 800,         # 较长输出
                "guidance_prefix": "Let me create a structured study plan:",
                "constraints": [
                    "Make the plan time-bound",
                    "Include specific milestones",
                    "Be realistic and achievable"
                ]
            },
            "brainstorm": {
                "temperature": 0.9,         # 高温度：创意头脑风暴
                "top_p": 0.99,
                "top_k": 60,
                "num_predict": 600,
                "guidance_prefix": "Here are several approaches to consider:",
                "constraints": [
                    "Generate diverse ideas",
                    "Be creative",
                    "Consider multiple perspectives"
                ]
            },
            "summarize": {
                "temperature": 0.0,         # 低温度：精确总结
                "top_p": 0.85,
                "top_k": 30,
                "num_predict": 300,         # 缩短输出
                "guidance_prefix": "Here is a concise summary:",
                "constraints": [
                    "Be comprehensive yet brief",
                    "Capture key points",
                    "Maintain accuracy"
                ]
            }
        }
        
        return params_map.get(mode, params_map["qa"])
    
    def get_generation_params(self) -> Dict:
        """获取ollama生成参数"""
        return {
            "temperature": self.params["temperature"],
            "top_p": self.params["top_p"],
            "top_k": self.params["top_k"],
            "num_predict": self.params["num_predict"]
        }
    
    def get_guidance_prefix(self) -> str:
        """获取生成引导前缀"""
        return self.params.get("guidance_prefix", "")
    
    def get_constraints(self) -> List[str]:
        """获取任务约束"""
        return self.params.get("constraints", [])
    
    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            "mode": self.mode,
            "temperature": self.params["temperature"],
            "top_p": self.params["top_p"],
            "top_k": self.params["top_k"],
            "num_predict": self.params["num_predict"],
            "guidance_prefix": self.params.get("guidance_prefix", ""),
            "constraints": self.params.get("constraints", [])
        }


# ============================================================
# Part 4: Enhanced Prompt Manager
# ============================================================

class PromptManager:
    """
    增强的提示词管理器
    集成：数据化定义、预算感知组装、生成控制与引导
    """
    
    def __init__(self, max_prompt_tokens: int = 3200):
        self.max_prompt_tokens = max_prompt_tokens
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """加载提示词模板"""
        return {
            "system_role": (
                "You are HKBU Study Companion, a friendly and accurate local assistant "
                "for Hong Kong Baptist University students. Your primary responsibility is to "
                "help students understand course content, create study plans, and answer academic questions."
            ),
            "context_instruction": (
                "CONTEXT USAGE RULES:\n"
                "• You MUST only use the provided context to answer questions\n"
                "• Always cite sources using [Source: title, Page X] format\n"
                "• If the context mentions specific course codes (e.g., COMP7045), provide comprehensive details\n"
                "• Never make up information outside the provided context\n"
                "• If information is missing or unclear, explicitly state this"
            ),
            "response_format": (
                "RESPONSE FORMAT:\n"
                "• Start with a direct answer\n"
                "• Support with evidence from context\n"
                "• End with relevant citations"
            ),
            "history_format": "PREVIOUS CONVERSATION:\n{content}",
            "query_format": "CURRENT QUESTION:\n{content}",
            "context_format": "RELEVANT CONTEXT:\n{content}"
        }
    
    def create_elements(
        self,
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa",
        include_history: bool = True
    ) -> List[PromptElement]:
        """
        创建提示词元素列表
        
        这实现了提示词元素的数据化定义
        """
        elements = []
        
        # 1. 系统角色（CRITICAL）
        elements.append(PromptElement(
            name="system_role",
            template=self.templates["system_role"],
            token_estimate=50,
            priority=ElementPriority.CRITICAL,
            required=True
        ))
        
        # 2. 上下文使用规则（HIGH）
        elements.append(PromptElement(
            name="context_rules",
            template=self.templates["context_instruction"],
            token_estimate=80,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        # 3. 响应格式要求（MEDIUM）
        elements.append(PromptElement(
            name="response_format",
            template=self.templates["response_format"],
            token_estimate=40,
            priority=ElementPriority.MEDIUM,
            required=False
        ))
        
        # 4. 上下文内容（HIGH）- 这是最重要的实际信息
        elements.append(PromptElement(
            name="context",
            template=self.templates["context_format"],
            content=context_str,
            token_estimate=len(context_str.split()) + 20,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        # 5. 对话历史（MEDIUM，可选）
        if include_history and history:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content'][:200]}..." 
                if len(m['content']) > 200 else f"{m['role'].upper()}: {m['content']}"
                for m in history[-4:]  # 只保留最近4条
            ])
            elements.append(PromptElement(
                name="history",
                template=self.templates["history_format"],
                content=history_text,
                token_estimate=len(history_text.split()) + 10,
                priority=ElementPriority.MEDIUM,
                required=False
            ))
        
        # 6. 当前查询（CRITICAL）
        elements.append(PromptElement(
            name="query",
            template=self.templates["query_format"],
            content=query,
            token_estimate=len(query.split()) + 10,
            priority=ElementPriority.CRITICAL,
            required=True
        ))
        
        return elements
    
    def assemble_prompt(
        self,
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa",
        token_budget: Optional[TokenBudget] = None
    ) -> Tuple[str, Dict]:
        """
        组装完整提示词
        
        这实现了预算感知的组装策略 + 生成控制与引导
        
        Args:
            query: 用户查询
            context_str: 检索到的上下文
            history: 对话历史
            mode: 任务模式（qa、plan、brainstorm、summarize）
            token_budget: token预算（可选）
        
        Returns:
            (assembled_prompt, metadata)
        """
        # 初始化预算
        if token_budget is None:
            token_budget = TokenBudget(
                total_budget=4096,
                reserved_for_output=800 if mode == "qa" else 1000
            )
        
        # 创建提示词元素
        elements = self.create_elements(
            query=query,
            context_str=context_str,
            history=history,
            mode=mode,
            include_history=len(history) > 0
        )
        
        # 使用组装策略
        strategy = PromptAssemblyStrategy(max_context_tokens=self.max_prompt_tokens)
        for elem in elements:
            strategy.add_element(elem)
        
        # 预算感知组装
        assembled_prompt, element_metadata = strategy.assemble_with_budget(token_budget)
        
        # 获取生成配置
        gen_config = GenerationConfig(mode=mode)
        
        # 添加生成引导
        guidance_prefix = gen_config.get_guidance_prefix()
        constraints = gen_config.get_constraints()
        
        constraint_text = "\nCONSTRAINTS:\n" + "\n".join([f"• {c}" for c in constraints])
        
        full_prompt = f"{assembled_prompt}{constraint_text}\n\n{guidance_prefix}\n"
        
        # 构建元数据
        metadata = {
            "mode": mode,
            "token_budget": token_budget.get_budget_info(),
            "element_metadata": element_metadata,
            "generation_config": gen_config.to_dict(),
            "total_prompt_length": len(full_prompt.split()),
            "constraints_applied": constraints
        }
        
        return full_prompt, metadata
    
    def assemble_prompt_simple(
        self,
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa"
    ) -> str:
        """
        简化版本：直接返回提示词（向后兼容）
        """
        prompt, _ = self.assemble_prompt(query, context_str, history, mode)
        return prompt
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        粗略估计文本的token数
        (实际应该使用tokenizer，这里用简单方法)
        """
        return len(text.split()) + 10
    
    @staticmethod
    def get_element_info() -> Dict:
        """获取所有可用元素的信息"""
        return {
            "element_priorities": {
                "CRITICAL": "必须包含，不能删除",
                "HIGH": "核心内容，尽量保留",
                "MEDIUM": "可选内容，可根据预算删除",
                "LOW": "辅助内容，优先删除"
            },
            "generation_modes": {
                "qa": "精确问答模式（低温度）",
                "plan": "创意规划模式（中温度）",
                "brainstorm": "头脑风暴模式（高温度）",
                "summarize": "精确总结模式（低温度）"
            }
        }