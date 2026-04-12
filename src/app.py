# HKBU Study Companion - Streamlit UI (Final Integrated Version)
import streamlit as st
import ollama
from pathlib import Path

# 导入后端模块
from rag_engine import RAGEngine
from prompt_manager import PromptManager
from chat_logic import ChatLogic
from react_engine import ReActEngine  # ReAct 推理引擎

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

# ====================== 初始化后端（只初始化一次） ======================
if "rag" not in st.session_state:
    with st.spinner("🚀 正在加载 RAG 引擎和知识库..."):
        # 默认使用 natural 切分（boundary-cut ratio 更低，推荐）
        st.session_state.rag = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
        st.session_state.pm = PromptManager()
        st.session_state.chat = ChatLogic(st.session_state.rag, st.session_state.pm)
        st.session_state.messages = []
        
        # 初始化 ReAct 引擎（但默认禁用）
        st.session_state.react_engine = ReActEngine(max_steps=5, model="gemma3:4b")
        st.session_state.react_enabled = False

st.title("📚 Study Companion")
st.caption("HKBU | Local AI Study Assistant powered by Ollama + RAG")

# ====================== 侧边栏设置 ======================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # 模型选择
    st.markdown("##### Model")
    model_name = st.selectbox(
        label="",
        options=["gemma3:4b", "gemma3:12b", "qwen3:9b"],
        index=0,
        label_visibility="collapsed"
    )
    
    # 检索方式
    st.markdown("##### Retrieval Mode")
    retrieval_method = st.selectbox(
        label="",
        options=["Embedding Search (Neural)", "Lexical Search"],
        index=0,
        label_visibility="collapsed"
    )
    retrieval_type = "neural" if "Neural" in retrieval_method else "lexical"
    
    # 生成参数
    st.markdown("##### Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0 if "qa" else 0.7, step=0.1)
    
    # ==================== ReAct 推理引擎 ====================
    st.markdown("##### 🧠 Reasoning Mode (ReAct)")
    react_enable = st.checkbox("Enable ReAct Reasoning", value=st.session_state.react_enabled)
    
    if react_enable != st.session_state.react_enabled:
        st.session_state.react_enabled = react_enable
        if react_enable:
            # 启用 ReAct
            st.session_state.chat.enable_react(st.session_state.react_engine)
            st.success("✅ ReAct Reasoning Enabled")
        else:
            # 禁用 ReAct
            st.session_state.chat.disable_react()
            st.info("ReAct Reasoning Disabled")
    
    if st.session_state.react_enabled:
        max_steps = st.slider(
            "Max Reasoning Steps",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of Thought-Action-Observation cycles"
        )
        st.session_state.react_engine.max_steps = max_steps
    
    # 文档管理
    st.markdown("##### Knowledge Base")
    if st.button("🔄 Rebuild Index (Natural)", use_container_width=True):
        with st.spinner("正在重建索引..."):
            st.session_state.rag = RAGEngine(chunk_file="chunks_natural_500_50.jsonl")
        st.success("索引已重建！")
    if st.button("🔄 Rebuild Index (Sliding)", use_container_width=True):
        with st.spinner("正在重建索引..."):
            st.session_state.rag = RAGEngine(chunk_file="chunks_sliding_500_50.jsonl")
        st.success("索引已重建！")
    
    st.info("📁 文档位于 `data/` 文件夹\n使用 `document_processor.py` 预处理")

# ====================== 主对话区 ======================
st.markdown("### 💬 Chat with Your Documents")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "cited_docs" in msg:
            st.caption(f"📌 Sources: {', '.join(msg['cited_docs'])}")

# 用户输入
user_input = st.chat_input("Ask anything about HKBU courses, policies, or study plans...")

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 调用后端处理
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chat.process_query(
                query=user_input,
                retrieval_type=retrieval_type
            )
        
        # 显示回答
        st.markdown(result["response"])
        
        # ==================== 显示 ReAct 推理过程（如果启用） ====================
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
        st.caption(f"📊 Tokens: {result['total_tokens']} | Mode: {result['mode'].upper()}")
    
    # 保存到历史（带引用信息）
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "cited_docs": result["cited_docs"]
    })

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
                plan_query = f"""
You are a professional study plan generator for HKBU students.

Create a realistic, step-by-step study plan based STRICTLY on the retrieved course materials.

Follow these rules:
1. ONLY use information from the provided course documents.
2. DO NOT invent content not in the documents.
3. DO NOT switch to other courses (e.g., do NOT use COMP7200 if the goal is COMP7045).
4. If information is insufficient, say "Not enough course material available."
5. Structure the plan clearly by time slots and topics.

Available time: {time_limit}
Study goal: {study_goal}
Intensity level: {intensity}

Generate the study plan now.
"""
                # 直接走同一个 RAG 管道（自动识别为 plan 模式）
                result = st.session_state.chat.process_query(
                    query=plan_query,
                    retrieval_type=retrieval_type,
                    use_react_override=False
                )
            
            st.success(f"🎯 Study Plan for: {study_goal}")
            st.markdown(result["response"])
            st.caption(f"📌 Sources: {', '.join(result['cited_docs'])}")
            
            # 同时加入聊天记录
            st.session_state.messages.append({"role": "user", "content": plan_query})
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "cited_docs": result["cited_docs"]
            })
        else:
            st.warning("⚠️ Please fill in Available Time and Study Goal.")

# ====================== LLM-as-a-Judge 评判区 ======================
st.markdown("### ⚖️ LLM-as-a-Judge ")
if st.session_state.messages:
    # 查找最近一条 assistant 回复和对应 user 问题
    last_user = None
    last_assistant = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg
        elif msg["role"] == "user" and last_user is None:
            last_user = msg
        if last_user and last_assistant:
            break
    if last_user and last_assistant:
        with st.expander("🔍 Judge the Last AI Answer"):
            if st.button("Run LLM Judge", key="judge_btn", use_container_width=True):
                with st.spinner("LLM is evaluating the answer..."):
                    # 获取上下文（可选：拼接最近检索内容）
                    context = ""
                    judge_result = st.session_state.chat.judge_response(
                        query=last_user["content"],
                        answer=last_assistant["content"],
                        context=context
                    )
                st.success(f"Score: {judge_result.get('score','?')}")
                st.markdown(f"**Reasoning:** {judge_result.get('reasoning','')}")
                st.markdown(f"**Suggestion:** {judge_result.get('suggestion','')}")
            else:
                st.info("Click the button to evaluate the latest AI response.")

# 页脚
st.caption("Built with Ollama + Local RAG | FSC 801CD Compatible")