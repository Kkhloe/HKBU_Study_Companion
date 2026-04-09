from .rag_engine import RAGEngine
from .prompt_manager import PromptManager
from .chat_logic import ChatLogic

def main():
    print("Initialize HKBU Study Companion...")
    
    rag = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")   # or sliding
    pm = PromptManager()
    chat = ChatLogic(rag, pm)

    print("\n HKBU Study Companion is ready！Input 'exit' quit\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            break

        result = chat.process_query(query, retrieval_type="neural")   # Change "lexical" for comparison.

        print(f"\nCompanion: {result['response']}")
        print(f" Cited: {result['cited_docs']}")
        print(f" Tokens: {result['total_tokens']} | Mode: {result['mode']}\n")