"""
Prompt Manager for HKBU Study Companion
Responsible for constructing high-quality prompts following best practices
Includes: dataified prompt element definitions, budget-aware assembly strategy, generation control
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json


# ============================================================
# Part 1: Prompt Element Data Model
# ============================================================

class ElementPriority(Enum):
    """Element priority levels for budget-constrained trimming"""
    CRITICAL = 4      # Cannot be deleted (system prompt, user query)
    HIGH = 3           # Core content, preserve when possible
    MEDIUM = 2         # Optional, can be deleted if needed
    LOW = 1             # Auxiliary, delete first


@dataclass
class PromptElement:
    """Dataified definition of a prompt element
    
    Attributes:
        name: Element name
        template: Element template (can contain {placeholders})
        content: Actual content or function that generates the element
        token_estimate: Estimated tokens for budget management
        priority: Priority level for trimming when tokens insufficient
        required: Whether this element must be included
        parameters: Element parameter dictionary
    """
    name: str
    template: str
    content: str = ""
    token_estimate: int = 0
    priority: ElementPriority = ElementPriority.MEDIUM
    required: bool = False
    parameters: Dict = field(default_factory=dict)
    
    def render(self, **kwargs) -> Tuple[str, int]:
        """Render element to text. Returns (rendered_text, estimated_tokens)"""
        try:
            rendered = self.template.format(content=self.content, **kwargs, **self.parameters)
        except (KeyError, IndexError):
            rendered = self.template
        
        estimated_tokens = len(rendered.split()) + 10
        return rendered, estimated_tokens


@dataclass
class TokenBudget:
    """Token budget manager
    
    Attributes:
        total_budget: Total token budget
        used_tokens: Tokens already used
        reserved_for_output: Tokens reserved for output
    """
    total_budget: int = 4096
    used_tokens: int = 0
    reserved_for_output: int = 800
    
    @property
    def available_for_prompt(self) -> int:
        """Available tokens for prompt"""
        return self.total_budget - self.reserved_for_output - self.used_tokens
    
    @property
    def remaining_budget(self) -> int:
        """Remaining budget"""
        return max(0, self.available_for_prompt)
    
    def allocate(self, tokens: int) -> bool:
        """Allocate tokens, return success status"""
        if self.used_tokens + tokens <= self.available_for_prompt:
            self.used_tokens += tokens
            return True
        return False
    
    def reset(self):
        """Reset budget"""
        self.used_tokens = 0
    
    def get_budget_info(self) -> Dict:
        """Get budget information"""
        return {
            "total": self.total_budget,
            "used": self.used_tokens,
            "available": self.available_for_prompt,
            "remaining": self.remaining_budget,
            "reserved_for_output": self.reserved_for_output
        }


# ============================================================
# Part 2: Prompt Assembly Strategy (Budget-aware)
# ============================================================

class PromptAssemblyStrategy:
    """Budget-aware prompt assembly strategy with flexible element selection and priority trimming"""
    
    def __init__(self, max_context_tokens: int = 1500):
        self.max_context_tokens = max_context_tokens
        self.elements: List[PromptElement] = []
    
    def add_element(self, element: PromptElement) -> None:
        """Add prompt element"""
        self.elements.append(element)
    
    def prioritize_elements(self) -> List[PromptElement]:
        """Sort elements by priority"""
        return sorted(self.elements, key=lambda e: e.priority.value, reverse=True)
    
    def assemble_with_budget(self, budget: TokenBudget) -> Tuple[str, Dict]:
        """Budget-aware assembly strategy. Returns (assembled_prompt, metadata)"""
        assembled_parts = []
        element_metadata = {
            "included_elements": [],
            "excluded_elements": [],
            "total_tokens_used": 0
        }
        
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
                if elem.required:
                    assembled_parts.append(rendered)
                    element_metadata["included_elements"].append({
                        "name": elem.name,
                        "tokens": tokens,
                        "priority": elem.priority.name,
                        "forced": True
                    })
        
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
# Part 3: Generation Control & Guidance
# ============================================================

class GenerationConfig:
    """Generation configuration for different task types"""
    
    def __init__(self, mode: str = "qa"):
        self.mode = mode
        self.params = self._get_default_params(mode)
    
    def _get_default_params(self, mode: str) -> Dict:
        """Get default generation parameters by task mode"""
        
        params_map = {
            "qa": {
                "temperature": 0.0,         # Low: precise answers
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 500,
                "guidance_prefix": "Based on the provided context, my answer is:",
                "constraints": [
                    "Must cite the source",
                    "Only use provided context",
                    "Be concise and accurate"
                ]
            },
            "plan": {
                "temperature": 0.7,         # Medium: creative planning
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 800,
                "guidance_prefix": "Let me create a structured study plan:",
                "constraints": [
                    "Make the plan time-bound",
                    "Include specific milestones",
                    "Be realistic and achievable"
                ]
            },
            "brainstorm": {
                "temperature": 0.9,         # High: creative brainstorm
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
                "temperature": 0.0,         # Low: precise summary
                "top_p": 0.85,
                "top_k": 30,
                "num_predict": 300,
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
        """Get ollama generation parameters"""
        return {
            "temperature": self.params["temperature"],
            "top_p": self.params["top_p"],
            "top_k": self.params["top_k"],
            "num_predict": self.params["num_predict"]
        }
    
    def get_guidance_prefix(self) -> str:
        """Get generation guidance prefix"""
        return self.params.get("guidance_prefix", "")
    
    def get_constraints(self) -> List[str]:
        """Get task constraints"""
        return self.params.get("constraints", [])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
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
    """Enhanced prompt manager integrating dataified definitions, budget-aware assembly, and generation control"""
    
    def __init__(self, max_prompt_tokens: int = 3200):
        self.max_prompt_tokens = max_prompt_tokens
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates"""
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
        """Create list of prompt elements"""
        elements = []
        
        elements.append(PromptElement(
            name="system_role",
            template=self.templates["system_role"],
            token_estimate=50,
            priority=ElementPriority.CRITICAL,
            required=True
        ))
        
        elements.append(PromptElement(
            name="context_rules",
            template=self.templates["context_instruction"],
            token_estimate=80,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        elements.append(PromptElement(
            name="response_format",
            template=self.templates["response_format"],
            token_estimate=40,
            priority=ElementPriority.MEDIUM,
            required=False
        ))
        
        elements.append(PromptElement(
            name="context",
            template=self.templates["context_format"],
            content=context_str,
            token_estimate=len(context_str.split()) + 20,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        if include_history and history:
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content'][:200]}..." 
                if len(m['content']) > 200 else f"{m['role'].upper()}: {m['content']}"
                for m in history[-4:]
            ])
            elements.append(PromptElement(
                name="history",
                template=self.templates["history_format"],
                content=history_text,
                token_estimate=len(history_text.split()) + 10,
                priority=ElementPriority.MEDIUM,
                required=False
            ))
        
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
        """Assemble complete prompt with budget-aware assembly and generation control
        
        Args:
            query: User query
            context_str: Retrieved context
            history: Conversation history
            mode: Task mode (qa, plan, brainstorm, summarize)
            token_budget: Token budget (optional)
        
        Returns:
            (assembled_prompt, metadata)
        """
        if token_budget is None:
            token_budget = TokenBudget(
                total_budget=4096,
                reserved_for_output=800 if mode == "qa" else 1000
            )
        
        elements = self.create_elements(
            query=query,
            context_str=context_str,
            history=history,
            mode=mode,
            include_history=len(history) > 0
        )
        
        strategy = PromptAssemblyStrategy(max_context_tokens=self.max_prompt_tokens)
        for elem in elements:
            strategy.add_element(elem)
        
        assembled_prompt, element_metadata = strategy.assemble_with_budget(token_budget)
        
        gen_config = GenerationConfig(mode=mode)
        
        guidance_prefix = gen_config.get_guidance_prefix()
        constraints = gen_config.get_constraints()
        
        constraint_text = "\nCONSTRAINTS:\n" + "\n".join([f"• {c}" for c in constraints])
        
        full_prompt = f"{assembled_prompt}{constraint_text}\n\n{guidance_prefix}\n"
        
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
        """Simple version: return prompt directly (backward compatible)"""
        prompt, _ = self.assemble_prompt(query, context_str, history, mode)
        return prompt
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (should use actual tokenizer in production)"""
        return len(text.split()) + 10
    
    @staticmethod
    def get_element_info() -> Dict:
        """Get information about all available elements"""
        return {
            "element_priorities": {
                "CRITICAL": "Must be included, cannot be deleted",
                "HIGH": "Core content, preserve when possible",
                "MEDIUM": "Optional content, can be deleted if needed",
                "LOW": "Auxiliary content, delete first"
            },
            "generation_modes": {
                "qa": "Precise QA mode (low temperature)",
                "plan": "Creative planning mode (medium temperature)",
                "brainstorm": "Brainstorming mode (high temperature)",
                "summarize": "Precise summarization mode (low temperature)"
            }
        }