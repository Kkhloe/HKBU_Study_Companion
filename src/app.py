# HKBU Study Companion - Streamlit UI (Final Integrated Version)
import streamlit as st
import ollama
from pathlib import Path
import re

# Backend modules
from rag_engine import RAGEngine
from prompt_manager import PromptManager
from chat_logic import ChatLogic
from react_engine import ReActEngine  # ReAct reasoning engine

st.set_page_config(
    page_title="HKBU Study Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS ======================
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
        
        # Initialize ReAct engine (disabled by default)
        st.session_state.react_engine = ReActEngine(max_steps=3, model="gemma3:4b")
        st.session_state.react_enabled = False

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
        options=["gemma3:4b", "gemma3:12b", "qwen3:8b"],
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
    
    # --- ReAct reasoning mode ---
    st.markdown("##### 🧠 Reasoning Mode (ReAct)")
    react_enable = st.checkbox("Enable ReAct Reasoning", value=st.session_state.react_enabled)
    
    if react_enable != st.session_state.react_enabled:
        st.session_state.react_enabled = react_enable
        if react_enable:
            # Enable ReAct
            st.session_state.chat.enable_react(st.session_state.react_engine)
            st.success("✅ ReAct Reasoning Enabled")
        else:
            # Disable ReAct
            st.session_state.chat.disable_react()
            st.info("ReAct Reasoning Disabled")
    
    if st.session_state.react_enabled:
        max_steps = st.slider(
            "Max Reasoning Steps",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of Thought-Action-Observation cycles"
        )
        st.session_state.react_engine.max_steps = max_steps
    
    # Knowledge base
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
                retrieval_type=retrieval_type,
                model=model_name,
            )
        
        st.markdown(result["response"])
        
        # Show ReAct steps (if enabled)
        if result.get("reasoning_enabled", False):
            with st.expander("🧠 Show Reasoning Steps"):
                for step in result.get("reasoning_steps", []):
                    st.write(f"**Step {step['step']}**")
                    st.write(f"Thought: {step['thought']}")
                    st.write(f"Action: {step['action']}")
                    st.write(f"Observation: {step['observation']}")
                    st.divider()
            
            if "reasoning_trace" in result:
                with st.expander("📋 Full Reasoning Trace"):
                    st.code(result["reasoning_trace"], language="text")
        
        st.caption(f"📌 Sources: {', '.join(result['cited_docs'])}")
        rq = result.get("retrieval_query", "")
        reuse = result.get("context_reused", False)
        extra = f" | Model: {result.get('model', model_name)}"
        if reuse:
            extra += " | Context: reused from last turn"
        elif rq and rq != user_input.strip():
            extra += f" | Search: {rq[:120]}{'…' if len(rq) > 120 else ''}"
        st.caption(f"📊 Tokens: {result['total_tokens']} | Mode: {result['mode'].upper()}{extra}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "cited_docs": result["cited_docs"]
    })
    
    sync_chat_history()

# --- Study plan generator ---
st.markdown("### 📅 Generate Study Plan")
with st.container():

    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit = st.text_input("Available Time", placeholder="e.g. 7 days, 2 hours/day")
    with col2:
        study_goal = st.text_input("Study Goal", placeholder="e.g. COMP4146 final project")
    with col3:
        intensity = st.selectbox("Intensity Level", ["Light", "Medium", "High"])

# Button and output should be full-width (not inside a column).
if st.button("🚀 Generate Personalized Study Plan", type="primary", use_container_width=True):
    if not (time_limit and study_goal):
        st.warning("⚠️ Please fill in Available Time and Study Goal.")
    else:
        # Extract and lock course code for retrieval.
        course_match = re.search(
            r"(COMP\s?\d{4}|DAAI\s?\d{4}|ITM\s?\d{4}|AIDM\s?\d{4})",
            study_goal,
            re.IGNORECASE,
        )
        forced_course_codes = []
        if course_match:
            course_code = course_match.group(1).upper().replace(" ", "")
            forced_course_codes = [course_code]
            st.info(f"🔒 Locked to course: {course_code}")

        with st.spinner("AI is creating your study plan using course documents..."):
            plan_query = f"""
You are a professional study plan generator for HKBU students.

Create a realistic, step-by-step study plan based STRICTLY on the retrieved course materials.

Follow these rules:
1. ONLY use information from the provided course documents.
2. DO NOT invent content not in the documents.
3. DO NOT switch to other courses. Use ONLY the target course materials.
4. If information is insufficient, say "Not enough course material available."
5. Structure the plan clearly by time slots and topics.

Available time: {time_limit}
Study goal: {study_goal}
Intensity level: {intensity}

Generate the study plan now.
"""
            sync_chat_history()

            
            result = st.session_state.chat.process_query(
                query=plan_query,
                retrieval_type="lexical",  # lexical retrieval for precise course matching
                use_react_override=st.session_state.react_enabled,
                model=model_name,
                user_time=time_limit,
                user_goals=study_goal,
                user_workload=intensity,
                course_codes=forced_course_codes,  # lock retrieval to this course
            )

        st.success(f"🎯 Study Plan for: {study_goal}")
        st.markdown(result["response"])

        
        if result.get("reasoning_enabled", False):
            with st.expander("🧠 Show Reasoning Steps"):
                for step in result.get("reasoning_steps", []):
                    st.write(f"**Step {step['step']}**")
                    st.write(f"Thought: {step['thought']}")
                    st.write(f"Action: {step['action']}")
                    st.write(f"Observation: {step['observation']}")
                    st.divider()
            if "reasoning_trace" in result:
                with st.expander("📋 Full Reasoning Trace"):
                    st.code(result["reasoning_trace"], language="text")

        st.caption(f"📌 Sources: {', '.join(result['cited_docs'])}")

        st.session_state.messages.append({"role": "user", "content": plan_query})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
                "cited_docs": result["cited_docs"],
            }
        )

        sync_chat_history()


# Footer
st.caption("Built with Ollama + Local RAG | FSC 801CD Compatible")