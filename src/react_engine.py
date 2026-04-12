"""
ReAct Engine for HKBU Study Companion
Modular design: independent reasoning module integrated with main logic via hooks
Core idea: Thought → Action → Observation → Iterative loop

Features:
- Fully decoupled, no modifications to existing ChatLogic, RAGEngine, etc.
- Interacts with external systems via callback hooks
- Supports optional enable/disable
"""

import ollama
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum


class ActionType(Enum):
    """Action types executable by ReAct"""
    SEARCH = "search"           # Retrieve documents
    ANALYZE = "analyze"         # Analyze information
    GENERATE = "generate"       # Generate response
    CLARIFY = "clarify"         # Clarify the query
    SYNTHESIZE = "synthesize"   # Synthesize information
    EXTERNAL_API = "external"   # Call external API


class ThinkingStep:
    """Single reasoning step"""
    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought = ""
        self.action = None  # ActionType
        self.action_input = ""
        self.observation = ""
        self.is_final = False
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action.value if self.action else None,
            "action_input": self.action_input,
            "observation": self.observation,
            "is_final": self.is_final
        }


class ReActEngine:
    """
    ReAct Reasoning Engine (Plugin Mode)
    
    Core workflow:
    1. Initial thought (Thought)
    2. Select action (Action) & action parameters (Action Input)
    3. Execute action to get observation (Observation)
    4. Iterate until conclusion is reached
    5. Return final answer
    """
    
    def __init__(self, 
                 max_steps: int = 5,
                 model: str = "gemma3:4b"):
        self.max_steps = max_steps
        self.model = model
        self.steps: List[ThinkingStep] = []
        
        # Registered action handlers (hooks)
        self.action_handlers: Dict[ActionType, Callable] = {}
        
        # Callback functions for interacting with main system
        self.on_search: Optional[Callable[[str], Tuple[str, List[Dict]]]] = None
        self.on_step: Optional[Callable[[ThinkingStep], None]] = None
        self.on_complete: Optional[Callable[[Dict], None]] = None

    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register custom action handler"""
        self.action_handlers[action_type] = handler

    def set_search_callback(self, callback: Callable[[str], Tuple[str, List[Dict]]]):
        """Set search callback (invokes RAG engine)"""
        self.on_search = callback

    def set_step_callback(self, callback: Callable[[ThinkingStep], None]):
        """Set step callback (for logging / UI updates)"""
        self.on_step = callback

    def set_complete_callback(self, callback: Callable[[Dict], None]):
        """Set completion callback"""
        self.on_complete = callback

    def _generate_thought(self, query: str, context: str, prev_steps: str = "") -> str:
        """Use LLM to generate thinking step"""
        prompt = f"""You are a reasoning assistant for HKBU Study Companion.
Analyze the user's query step by step.

Query: {query}
Context: {context}

{f"Previous steps: {prev_steps}" if prev_steps else ""}

Provide your thought on what needs to be done next. Be concise."""
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 200,
                "top_p": 0.9,
            }
        )
        return response["response"].strip()

    def _parse_action(self, llm_output: str) -> Tuple[Optional[ActionType], str]:
        """Parse action and parameters from LLM output"""
        llm_lower = llm_output.lower()
        
        # Pattern matching (simplified)
        if "search" in llm_lower or "retrieve" in llm_lower or "find" in llm_lower:
            action_type = ActionType.SEARCH
            # Extract search keywords
            action_input = llm_output.split("search")[-1].strip() if "search" in llm_lower else llm_output[:100]
        elif "analyze" in llm_lower or "examine" in llm_lower:
            action_type = ActionType.ANALYZE
            action_input = llm_output[:100]
        elif "generate" in llm_lower or "write" in llm_lower or "create" in llm_lower:
            action_type = ActionType.GENERATE
            action_input = llm_output[:100]
        elif "final" in llm_lower or "answer" in llm_lower or "conclude" in llm_lower:
            action_type = ActionType.SYNTHESIZE
            action_input = llm_output[:500]
        else:
            action_type = ActionType.ANALYZE
            action_input = llm_output[:100]
        
        return action_type, action_input

    def _execute_action(self, action: ActionType, action_input: str, query: str) -> str:
        """Execute specified action and return observation"""
        
        if action == ActionType.SEARCH and self.on_search:
            # Invoke external search callback
            context, chunks = self.on_search(action_input)
            return f"Found relevant documents:\n{context[:300]}..."
        
        elif action == ActionType.ANALYZE:
            return f"Analyzed: {action_input[:200]}. More context needed."
        
        elif action == ActionType.SYNTHESIZE:
            return f"Synthesized conclusion: {action_input[:300]}"
        
        elif action in self.action_handlers:
            return self.action_handlers[action](action_input)
        
        else:
            return f"Action {action.value} not yet implemented."

    def reason(self, query: str, context: str = "", retrieval_callback: Optional[Callable] = None) -> Dict:
        """
        Core reasoning loop
        
        Parameters:
        - query: user question
        - context: initial context
        - retrieval_callback: RAG search callback
        
        Returns:
        {
            "final_answer": str,
            "reasoning_steps": List[Dict],
            "total_steps": int,
            "success": bool
        }
        """
        
        # Set search callback if provided
        if retrieval_callback:
            self.set_search_callback(retrieval_callback)
        
        self.steps = []
        prev_steps_summary = ""
        
        for step_num in range(1, self.max_steps + 1):
            step = ThinkingStep(step_num)
            
            # 1. Generate thought
            step.thought = self._generate_thought(query, context, prev_steps_summary)
            
            # 2. Parse action
            step.action, step.action_input = self._parse_action(step.thought)
            
            # 3. Execute action
            step.observation = self._execute_action(step.action, step.action_input, query)
            
            # 4. Check if conclusion is reached
            if step.action == ActionType.SYNTHESIZE or "final" in step.thought.lower():
                step.is_final = True
                self.steps.append(step)
                if self.on_step:
                    self.on_step(step)
                break
            
            self.steps.append(step)
            if self.on_step:
                self.on_step(step)
            
            # Accumulate step info for next round
            prev_steps_summary += f"\nStep {step_num}: {step.thought[:100]}"
        
        # Build final answer
        if self.steps:
            final_step = self.steps[-1]
            final_answer = final_step.observation if final_step.is_final else self._synthesize_answer()
        else:
            final_answer = "No reasoning steps completed."
        
        result = {
            "final_answer": final_answer,
            "reasoning_steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps),
            "success": len(self.steps) > 0,
            "query": query
        }
        
        if self.on_complete:
            self.on_complete(result)
        
        return result

    def _synthesize_answer(self) -> str:
        """Synthesize final answer from all steps"""
        if not self.steps:
            return "Unable to generate answer."
        
        observations = [s.observation for s in self.steps]
        prompt = f"""Based on the following reasoning steps, provide a final comprehensive answer:

{chr(10).join(observations)}

Final Answer:"""
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.5,
                "num_predict": 500,
            }
        )
        return response["response"].strip()

    def get_reasoning_trace(self) -> str:
        """Get human-readable reasoning process"""
        trace = f"\n{'='*60}\nReAct Reasoning Trace\n{'='*60}\n"
        for step in self.steps:
            trace += f"\n[Step {step.step_num}]\n"
            trace += f"Thought: {step.thought}\n"
            trace += f"Action: {step.action.value if step.action else 'None'}\n"
            trace += f"Observation: {step.observation[:200]}...\n"
        trace += f"\n{'='*60}\n"
        return trace

    def reset(self):
        """Reset engine state"""
        self.steps = []