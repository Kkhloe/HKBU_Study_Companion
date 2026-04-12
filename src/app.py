# HKBU Study Companion - Streamlit UI (Final Integrated Version)
import streamlit as st
import ollama
from pathlib import Path

from src.rag_engine import RAGEngine
from src.prompt_manager import PromptManager
from src.chat_logic import ChatLogic

st.set_page_config(
    page_title="HKBU Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS (保持您原有的精致风格) ======================
st.markdown("""
<style>
* {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 300;}
#MainMenu, footer, header {visibility: hidden;}
.stApp {background-color: #FFFFFF;}
div[data-testid="stChatMessage"], div[data-testid="stExpander"], div.stButton > button {
    border-radius: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: none;
}
div.stButton > button {background-color: #F5F5F7; color: #1D1D1F; padding: 8px 16px;}
div.stButton > button:hover {background-color: #E1E1E2;}
.stTextInput > div > div, .stChatInputContainer {border-radius: 16px; border: 1px solid #F2F2F2;}
section[data-testid="stSidebar"] {background-color: #FAFAFA; border-right: 1px solid #F2F2F2;}
h1 {font-weight: 400; color: #1D1D1F; letter-spacing: -0.5px;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

if "rag" not in st.session_state:
    with st.spinner("🚀 Loading RAG Engine and Knowledge Base..."):
        st.session_state.rag = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
        st.session_state.pm = PromptManager()
        st.session_state.chat = ChatLogic(st.session_state.rag, st.session_state.pm)
        st.session_state.messages = []

def sync_chat_history():
    """Sync Streamlit UI messages with ChatLogic internal history"""
    st.session_state.chat.history.clear()
    
    for msg in st.session_state.messages:
        st.session_state.chat.history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

sync_chat_history()

st.title("📚 Study Companion")
st.caption("HKBU | Local AI Study Assistant powered by Ollama + RAG")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("##### Model")
    model_name = st.selectbox(
        label="",
        options=["gemma3:4b", "gemma3:12b", "qwen3:9b"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("##### Retrieval Mode")
    retrieval_method = st.selectbox(
        label="",
        options=["Embedding Search (Neural)", "Lexical Search"],
        index=0,
        label_visibility="collapsed"
    )
    retrieval_type = "neural" if "Neural" in retrieval_method else "lexical"
    
    st.markdown("##### Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0 if "qa" else 0.7, step=0.1)
    
    st.markdown("##### Knowledge Base")
    if st.button("🔄 Rebuild Index (Natural)", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            st.session_state.rag = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
        st.success("Index rebuilt!")
    if st.button("🔄 Rebuild Index (Sliding)", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            st.session_state.rag = RAGEngine(chunk_file="chunks_sliding_500_50.jsonl")
        st.success("Index rebuilt!")
    
    st.info("📁 Documents in `data/` folder. Use `document_processor.py` to preprocess.")

st.markdown("### 💬 Chat with Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "cited_docs" in msg:
            st.caption(f"📌 Sources: {', '.join(msg['cited_docs'])}")

user_input = st.chat_input("Ask anything about HKBU courses, policies, or study plans...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    sync_chat_history()
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chat.process_query(
                query=user_input,
                retrieval_type=retrieval_type
            )
        
        st.markdown(result["response"])
        st.caption(f"📌 Sources: {', '.join(result['cited_docs'])}")
        st.caption(f"📊 Tokens: {result['total_tokens']} | Mode: {result['mode'].upper()}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "cited_docs": result["cited_docs"]
    })
    
    sync_chat_history()

# ====================== 学习计划生成器 ======================
st.markdown("### 📅 Generate Study Plan")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit = st.text_input("Available Time", placeholder="e.g. 7 days, 2 hours/day")
    with col2:
        study_goal = st.text_input("Study Goal", placeholder="e.g. COMP4146 final project")
    with col3:
        intensity = st.selectbox("Intensity Level", ["Light", "Medium", "High"])
    
    if st.button("🚀 Generate Personalized Study Plan", type="primary", use_container_width=True):
        if time_limit and study_goal:
            with st.spinner("AI is creating your study plan using course documents..."):
                plan_query = (
                    f"Create a realistic study plan. "
                    f"Available time: {time_limit}. "
                    f"Goal: {study_goal}. "
                    f"Intensity: {intensity}."
                )
                
                # 【关键修复】一样要同步历史
                sync_chat_history()
                
                # 直接走同一个 RAG 管道（自动识别为 plan 模式）
                result = st.session_state.chat.process_query(
                    query=plan_query,
                    retrieval_type=retrieval_type
                )
            
            st.success(f"🎯 Study Plan for: {study_goal}")
            st.markdown(result["response"])
            st.caption(f"📌 Sources: {', '.join(result['cited_docs'])}")
            
            st.session_state.messages.append({"role": "user", "content": plan_query})
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "cited_docs": result["cited_docs"]
            })
            
            sync_chat_history()
        else:
            st.warning("⚠️ Please fill in Available Time and Study Goal.")

st.caption("Built with Ollama + Local RAG | FSC 801CD Compatible")