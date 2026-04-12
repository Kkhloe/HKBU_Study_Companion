"""
Judge Engine for LLM-as-a-Judge
Use "gemma3:12b" to evaluate the quality of generated answers.
"""
import ollama
from typing import Dict

class JudgeEngine:
    def __init__(self, model: str = "gemma3:4b"):
        self.model = model

    def judge(self, query: str, answer: str, context: str = "") -> Dict:
        """
        Use LLM to judge the quality of an answer given a query and (optionally) context.
        Returns a dict with score, reasoning, and suggestion.
        """
        judge_prompt = f"""
You are an impartial academic judge. Given the following user question, answer, and context, evaluate the answer's correctness, completeness, and faithfulness to the context. 

Question: {query}
Answer: {answer}
Context: {context}

Please provide:
1. A score from 1 (poor) to 5 (excellent)
2. A brief reasoning (2-3 sentences)
3. Suggestions for improvement (if any)

Format:
Score: <number>
Reasoning: <text>
Suggestion: <text>
"""
        response = ollama.generate(
            model=self.model,
            prompt=judge_prompt,
            options={"temperature": 0.0, "num_predict": 256}
        )
        return self._parse_judge_response(response["response"]) if "response" in response else {}

    def _parse_judge_response(self, text: str) -> Dict:
        # Simple parser for the expected format
        lines = text.strip().split("\n")
        result = {"score": None, "reasoning": "", "suggestion": ""}
        for line in lines:
            if line.startswith("Score:"):
                try:
                    result["score"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    result["score"] = None
            elif line.startswith("Reasoning:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("Suggestion:"):
                result["suggestion"] = line.split(":", 1)[1].strip()
        return result
