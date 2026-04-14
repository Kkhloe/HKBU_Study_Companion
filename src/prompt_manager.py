"""
Prompt Manager for HKBU Study Companion
Responsible for constructing high-quality prompts following best practices
Includes: dataified prompt element definitions, budget-aware assembly strategy, generation control
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json
import re


# ============================================================
# Part 1: Prompt Element Data Model
# ============================================================

class ElementPriority(Enum):
    CRITICAL = 4      # system prompt, user query
    HIGH = 3         
    MEDIUM = 2        
    LOW = 1            


@dataclass
class PromptElement:
    name: str
    template: str
    content: str = ""
    token_estimate: int = 0
    priority: ElementPriority = ElementPriority.MEDIUM
    required: bool = False
    parameters: Dict = field(default_factory=dict)
    
    def render(self, **kwargs) -> Tuple[str, int]:
        try:
            rendered = self.template.format(content=self.content, **kwargs, **self.parameters)
        except (KeyError, IndexError):
            rendered = self.template
        
        estimated_tokens = len(rendered.split()) + 10
        return rendered, estimated_tokens


@dataclass
class TokenBudget:
    total_budget: int = 4096
    used_tokens: int = 0
    reserved_for_output: int = 800
    
    @property
    def available_for_prompt(self) -> int:
        return self.total_budget - self.reserved_for_output - self.used_tokens
    
    @property
    def remaining_budget(self) -> int:
        return max(0, self.available_for_prompt)
    
    def allocate(self, tokens: int) -> bool:
        if self.used_tokens + tokens <= self.available_for_prompt:
            self.used_tokens += tokens
            return True
        return False
    
    def reset(self):
        self.used_tokens = 0
    
    def get_budget_info(self) -> Dict:
        return {
            "total": self.total_budget,
            "used": self.used_tokens,
            "available": self.available_for_prompt,
            "remaining": self.remaining_budget,
            "reserved_for_output": self.reserved_for_output
        }


# ============================================================
# Part 2: Prompt Assembly Strategy 
# ============================================================

class PromptAssemblyStrategy:
    def __init__(self, max_context_tokens: int = 1500):
        self.max_context_tokens = max_context_tokens
        self.elements: List[PromptElement] = []
    
    def add_element(self, element: PromptElement) -> None:
        self.elements.append(element)
    
    def prioritize_elements(self) -> List[PromptElement]:
        return sorted(self.elements, key=lambda e: e.priority.value, reverse=True)
    
    def assemble_with_budget(self, budget: TokenBudget) -> Tuple[str, Dict]:
        assembled_parts = []
        element_metadata = {
            "included_elements": [],
            "excluded_elements": [],
            "total_tokens_used": 0
        }
        
        # Process critical/required elements first
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
                # Force include if marked as required
                if elem.required:
                    assembled_parts.append(rendered)
                    element_metadata["included_elements"].append({
                        "name": elem.name,
                        "tokens": tokens,
                        "priority": elem.priority.name,
                        "forced": True
                    })
        
        # Process optional elements in priority order
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
    def __init__(self, mode: str = "qa"):
        self.mode = mode
        self.params = self._get_default_params(mode)
    
    def _get_default_params(self, mode: str) -> Dict:
        params_map = {
            "qa": {
                "temperature": 0.0,
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
                "temperature": 0.3,
                "top_p": 0.85,
                "top_k": 30,
                "num_predict": 1200,
                "guidance_prefix": "Let me create a detailed, day-by-day study plan using the specific course content:",
                "constraints": [
                    "CONSISTENCY: Use identical time format across ALL days",
                    "CONTENT MAP: Extract all core topics from the context",
                    "DAILY ALLOCATION: Evenly distribute core content across days",
                    "SPECIFICITY: Reference actual chapters/sections/topics",
                    "PROGRESSION: Fundamentals first, then advanced concepts",
                    "CONCRETE TASKS: Specify exact materials and practice",
                    "TERMINOLOGY: Use only course-specific terms",
                    "PRACTICE: Include mentioned exercises/assessments",
                    "HEADERS: Use 'Day N (X hours)' format consistently"
                ]
            },
            "brainstorm": {
                "temperature": 0.9,
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
                "temperature": 0.0,
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
        return {
            "temperature": self.params["temperature"],
            "top_p": self.params["top_p"],
            "top_k": self.params["top_k"],
            "num_predict": self.params["num_predict"]
        }
    
    def get_guidance_prefix(self) -> str:
        return self.params.get("guidance_prefix", "")
    
    def get_constraints(self) -> List[str]:
        return self.params.get("constraints", [])
    
    def to_dict(self) -> Dict:
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
    def __init__(self, max_prompt_tokens: int = 3200):
        self.max_prompt_tokens = max_prompt_tokens
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
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
                "• Provide full details for mentioned course codes\n"
                "• Never make up information outside the context\n"
                "• State clearly if information is missing\n"
                "• When answering about a specific course, DO NOT include content from other courses"
            ),
            "plan_instruction": (
                "STUDY PLAN REQUIREMENTS:\n"
                "1. TIME FORMAT CONSISTENCY: Use uniform time units\n"
                "2. CORE CONTENT EXTRACTION: Identify all key topics from context\n"
                "3. DAILY ALLOCATION: Distribute content evenly across available days\n"
                "4. SPECIFIC LEARNING ASSIGNMENTS: List exact chapters, pages, tasks\n"
                "5. FORMATTING: Use clear, consistent structure"
            ),
            "response_format": (
                "RESPONSE FORMAT:\n"
                "• Start with a direct answer\n"
                "• Support with evidence from context\n"
                "• End with relevant citations"
            ),
            "history_format": "PREVIOUS CONVERSATION:\n{content}",
            "query_format": "CURRENT QUESTION:\n{content}",
            "context_format": "RELEVANT CONTEXT:\n{content}",
            "user_constraints": "USER CONSTRAINTS:\n{content}"
        }
    
    def create_elements(
        self,
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa",
        include_history: bool = True,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None
    ) -> List[PromptElement]:
        
        elements = []
        
        # System role (critical)
        elements.append(PromptElement(
            name="system_role",
            template=self.templates["system_role"],
            token_estimate=50,
            priority=ElementPriority.CRITICAL,
            required=True
        ))
        
        # Context usage rules (high priority)
        elements.append(PromptElement(
            name="context_rules",
            template=self.templates["context_instruction"],
            token_estimate=80,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        # Add study plan specific rules if in plan mode
        if mode == "plan":
            elements.append(PromptElement(
                name="plan_rules",
                template=self.templates["plan_instruction"],
                token_estimate=100,
                priority=ElementPriority.HIGH,
                required=True
            ))
        
        # Response format guidance
        elements.append(PromptElement(
            name="response_format",
            template=self.templates["response_format"],
            token_estimate=40,
            priority=ElementPriority.MEDIUM,
            required=False
        ))
        
        # Retrieved context (required)
        elements.append(PromptElement(
            name="context",
            template=self.templates["context_format"],
            content=context_str,
            token_estimate=len(context_str.split()) + 20,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        # User constraints (time, goals, workload)
        if user_time or user_goals or user_workload:
            constraints_parts = []
            if user_time:
                constraints_parts.append(f"Available time: {user_time}")
            if user_goals:
                constraints_parts.append(f"Goals: {user_goals}")
            if user_workload:
                constraints_parts.append(f"Current workload: {user_workload}")
            
            constraints_text = "\n".join(constraints_parts)
            elements.append(PromptElement(
                name="user_constraints",
                template=self.templates["user_constraints"],
                content=constraints_text,
                token_estimate=len(constraints_text.split()) + 10,
                priority=ElementPriority.HIGH if mode == "plan" else ElementPriority.MEDIUM,
                required=False
            ))
        
        # Content mapping guidance for study plans
        if mode == "plan":
            content_mapping_guidance = (
                "CONTENT MAPPING FOR STUDY PLAN:\n"
                "1. List core topics from context\n"
                "2. Order from basic to advanced\n"
                "3. Distribute evenly across days\n"
                "4. Specify exact sections/pages\n"
                "Example: Day 1 (3 hours): Chapter 2 (pages 10-35) + Exercises"
            )
            elements.append(PromptElement(
                name="content_mapping",
                template=content_mapping_guidance,
                content="",
                token_estimate=120,
                priority=ElementPriority.HIGH,
                required=True
            ))
        
        # Conversation history
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
        
        # User query (critical)
        elements.append(PromptElement(
            name="query",
            template=self.templates["query_format"],
            content=query,
            token_estimate=len(query.split()) + 10,
            priority=ElementPriority.CRITICAL,
            required=True
        ))
        
        # ============================================================
        # Auto prerequisite instruction (no mapping table needed)
        # Instruct AI to mention prerequisites stated in context
        # ============================================================
        prerequisite_note = (
            "IMPORTANT:\n"
            "If the context mentions course prerequisites, you MUST mention them briefly at the END of your answer.\n"
            "Do NOT explain prerequisite content — only state that prior knowledge is required."
        )
        elements.append(PromptElement(
            name="prerequisite_notice",
            template=prerequisite_note,
            token_estimate=40,
            priority=ElementPriority.HIGH,
            required=True
        ))
        
        return elements
    
    def assemble_prompt(
        self,
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa",
        token_budget: Optional[TokenBudget] = None,
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None
    ) -> Tuple[str, Dict]:
        
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
            include_history=len(history) > 0,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload
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
        mode: str = "qa",
        user_time: Optional[str] = None,
        user_goals: Optional[str] = None,
        user_workload: Optional[str] = None
    ) -> str:
        
        prompt, _ = self.assemble_prompt(
            query, context_str, history, mode,
            user_time=user_time,
            user_goals=user_goals,
            user_workload=user_workload
        )
        return prompt
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text.split()) + 10
    
    @staticmethod
    def get_element_info() -> Dict:
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