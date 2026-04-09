from src.rag_engine import RAGEngine
from src.prompt_manager import PromptManager
from src.chat_logic import ChatLogic

rag = RAGEngine()
rag.load_index()
pm = PromptManager()
chat = ChatLogic(rag, pm)
print("Assess environmental readiness")

##----------------Evaluation script---------------##
test_queries = [
    "What is the group registration deadline for COMP4146?",
    "What is the late submission policy?",
    "Help me make a 2-week study plan for the Prompt Engineering final project while taking 3 courses."
]

print("=== BASELINE EVALUATION (No-RAG vs RAG) ===\n")
for q in test_queries:
    print(f"Query: {q}")
    # No-RAG
    no_rag_prompt = f"You are HKBU Study Companion.\nUser: {q}\nAssistant: "
    no_rag = ollama.generate(model="gemma3:4b", prompt=no_rag_prompt, options={"temperature": 0.0, "num_predict": 600})
    print(f"  No-RAG  → Tokens: {no_rag.get('prompt_eval_count',0)+no_rag.get('eval_count',0)}")

    # RAG Neural
    res_n = chat.process_query(q, "neural")
    print(f"  RAG-Neural → Tokens: {res_n['total_tokens']} | Cited: {res_n['cited_docs'][:2]}...")

    # RAG Lexical
    res_l = chat.process_query(q, "lexical")
    print(f"  RAG-Lexical → Tokens: {res_l['total_tokens']}")
    print("-" * 80)
