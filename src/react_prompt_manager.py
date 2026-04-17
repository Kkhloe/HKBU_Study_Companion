"""
ReAct Prompt Manager - Dedicated Prompt Generator for ReAct Reasoning
Same interface as the standard PromptManager, but optimized for ReAct workflow prompt structure
Currently not calling ReactPromptManager.
For advanced prompt engineering, you can replace the three inline methods in react_engine.py with calls to ReactPromptManager:
    _generate_thought() → ReactPromptManager.assemble_react_thought_prompt()
    _parse_action() → ReactPromptManager.assemble_react_action_prompt()
    _synthesize_answer() → ReactPromptManager.assemble_react_synthesis_prompt()
"""

from typing import List, Dict


class ReactPromptManager:
    @staticmethod
    def assemble_react_thought_prompt(
        query: str,
        context_str: str,
        step_num: int = 1,
        prev_steps: str = ""
    ) -> str:
        
        prompt = f"""You are an intelligent reasoning assistant for HKBU Study Companion.
Analyze the user's question step by step. Think carefully about what needs to be done.

Current Step: {step_num}

User Question: {query}

Available Context:
{context_str}

{f"Previous reasoning steps:{chr(10)}{prev_steps}" if prev_steps else ""}

Think about:
1. What information is already available?
2. What is missing?
3. What action should I take next? (search, analyze, generate, or conclude?)

Your Thought:"""
        
        return prompt

    @staticmethod
    def assemble_react_action_prompt(
        query: str,
        thought: str,
        available_actions: List[str] = None
    ) -> str:
        
        if available_actions is None:
            available_actions = ["search", "analyze", "clarify", "conclude"]
        
        actions_str = "\n".join([f"- {a}" for a in available_actions])
        
        prompt = f"""Based on your thought, what action should you take?

Question: {query}

Your Thought: {thought}

Available Actions:
{actions_str}

Select one action and provide the input for that action.

Format:
Action: [action_name]
Action Input: [specific input or query for the action]

Your Response:"""
        
        return prompt

    @staticmethod
    def assemble_react_synthesis_prompt(
        query: str,
        reasoning_steps: List[Dict],
        context: str
    ) -> str:
        
        # Build summary of reasoning steps
        steps_summary = ""
        for i, step in enumerate(reasoning_steps, 1):
            steps_summary += f"\n[Step {i}]\n"
            steps_summary += f"Thought: {step.get('thought', '')[:100]}\n"
            steps_summary += f"Action: {step.get('action', 'N/A')}\n"
            steps_summary += f"Observation: {step.get('observation', '')[:150]}...\n"
        
        prompt = f"""Based on the following reasoning process, provide a comprehensive final answer.

Original Question: {query}

Reasoning Process:
{steps_summary}

Relevant Context:
{context[:500]}

Now provide a clear, well-structured final answer that:
1. Addresses the user's question directly
2. Cites the context where applicable
3. Is accurate and complete

Final Answer:"""
        
        return prompt

    @staticmethod
    def assemble_react_refinement_prompt(
        draft_answer: str,
        query: str,
        context: str
    ) -> str:
        
        prompt = f"""Review and refine the following answer for accuracy and completeness.

Original Question: {query}

Draft Answer:
{draft_answer}

Context Used:
{context[:300]}

Improvements to make (if any):
- Add missing information?
- Clarify ambiguous points?
- Better structure?
- Ensure all claims are supported by context?

Refined Final Answer:"""
        
        return prompt