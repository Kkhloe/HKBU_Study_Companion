"""
Prompt Manager for HKBU Study Companion
负责构建高质量提示词，完全符合课程 Topic 6 最佳实践
"""

from typing import List, Dict

class PromptManager:
    @staticmethod
    def assemble_prompt(
        query: str,
        context_str: str,
        history: List[Dict],
        mode: str = "qa"
    ) -> str:
        """
        Construct a complete prompt word structure that meets the course requirements:
        1. Introduction
        2. Context Parade(With citation)
        3. Refocus(Force context-only usage + must reference)
        4. Conversation History
        5. Transition + Inception（(Force the model into thinking mode)
        """
        # 1. Introduction
        intro = (
            "You are HKBU Study Companion, a friendly and accurate local assistant "
            "for Hong Kong Baptist University students. You ONLY use the provided context."
        )

        # 2. Context Parade
        context_section = f"=== RELEVANT CONTEXT ===\n{context_str}\n"

        # 3. Refocus(Strict constraints)
        refocus = (
            "Rules:\n"
            "• You are answering a question about a SPECIFIC course.\n"
            "• If the context mentions the course code (e.g. COMP7045), summarize its full content, topics, objectives, and assessment.\n"
            "• Always start with the official course title if available.\n"
            "• Never say only the title. Provide detailed explanation using the retrieved context.\n"
            "• Always include citation tags.\n"
            "• If information is missing, say so clearly."
        )

        # 4. History
        if history:
            history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
        else:
            history_str = "No previous messages."

        history_section = f"=== CONVERSATION HISTORY ===\n{history_str}\n"

        # 5. Transition + Inception
        if mode == "plan":
            transition = (
                f"User request: {query}\n"
                "Create a realistic, time-bound study plan based on the context and user's constraints.\n"
                "Assistant: Let me think step by step and create a clear weekly schedule...\n"
            )
        else:
            transition = f"User question: {query}\nAssistant: "

        full_prompt = f"""{intro}

{context_section}
{refocus}

{history_section}
{transition}"""

        return full_prompt