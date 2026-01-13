import streamlit as st
import os
import sys
import redis
import base64
from typing_extensions import TypedDict

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_community.cache import RedisCache

# 🔥 1. 混合检索模块检测
HYBRID_SEARCH_AVAILABLE = False
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    import rank_bm25
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    pass

# 🔥 2. LangChain 版本适配
try:
    from langchain.globals import set_llm_cache
except ImportError:
    import langchain
    def set_llm_cache(cache):
        langchain.llm_cache = cache

# ==========================================
# 🎨 UI & 样式配置 (修改了标题)
# ==========================================
st.set_page_config(
    page_title="Knowledge Q&A Bot",  # 🟢 改动点 1：浏览器标签页标题
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 定义生成头像的函数 ---
def get_icon_base64(color_hex):
    """生成纯色 SVG 头像的 Base64 字符串"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="50" fill="{color_hex}" /></svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# --- 2. 生成具体的头像数据 ---
# 天空蓝 (User)
USER_COLOR = "#87CEEB"
user_b64 = get_icon_base64(USER_COLOR)
USER_AVATAR = f"data:image/svg+xml;base64,{user_b64}"

# 淡粉色 (Bot)
BOT_COLOR = "#FFB6C1"
bot_b64 = get_icon_base64(BOT_COLOR)
BOT_AVATAR = f"data:image/svg+xml;base64,{bot_b64}"

# --- 3. 动态注入 CSS (左右对话气泡) ---
st.markdown(f"""
<style>
    /* 隐藏顶部菜单 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* 聊天气泡基础样式 */
    [data-testid="stChatMessage"] {{
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }}
    
    /* 🔥 用户消息 (靠右，蓝色背景) */
    [data-testid="stChatMessage"]:has(img[src="{USER_AVATAR}"]) {{
        flex-direction: row-reverse;
        background-color: rgba(135, 206, 235, 0.15);
        border: 1px solid {USER_COLOR};
        text-align: right;
    }}
    
    /* 🔥 机器人消息 (靠左，粉色背景) */
    [data-testid="stChatMessage"]:has(img[src="{BOT_AVATAR}"]) {{
        background-color: rgba(255, 182, 193, 0.15);
        border: 1px solid {BOT_COLOR};
    }}
    
    /* 调整头像大小 */
    [data-testid="stChatMessageAvatar"] img {{
        width: 45px;
        height: 45px;
    }}
</style>
""", unsafe_allow_html=True)

# 配置 API Key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ==========================================
# 🚀 核心系统初始化
# ==========================================
@st.cache_resource
def initialize_system():
    status = {"redis": False, "hybrid": False}
    
    try:
        client = redis.Redis(host="localhost", port=6379, db=0)
        try:
            set_llm_cache(RedisCache(redis_client=client))
        except TypeError:
            set_llm_cache(RedisCache(redis_=client))  
        status["redis"] = True
    except Exception:
        pass 

    if not os.path.exists("manual_parsed.md"):
        st.error("❌ Critical Error: 'manual_parsed.md' file missing!")
        st.stop()

    loader = UnstructuredMarkdownLoader("manual_parsed.md")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="manual_rag_web", 
        persist_directory="./chroma_db_web"
    )
    
    # 检索器设置 (k=6)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    if HYBRID_SEARCH_AVAILABLE:
        try:
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 6
            ensemble_retriever = EnsembleRetriever(
                retrievers=[base_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            retriever = ensemble_retriever
            status["hybrid"] = True
        except Exception:
            retriever = base_retriever
    else:
        retriever = base_retriever
        
    return retriever, status

with st.spinner('🚀 Booting Knowledge Q&A Bot...'):
    retriever, system_status = initialize_system()

# ==========================================
# 🎨 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ System Status")
    
    if system_status["redis"]:
        st.success("✅ Redis: On")
    else:
        st.warning("⚠️ Redis: Off")
    
    if system_status["hybrid"]:
        st.success("✅ Hybrid Search: On")
    else:
        st.info("ℹ️ Vector Search Only")
    
    st.divider()
    st.caption("Knowledge Source: manual_parsed.md")

# ==========================================
# 🧠 Agent 逻辑
# ==========================================
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve(state: GraphState):
    documents = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in documents])
    return {"context": context}

def generate(state: GraphState):
    prompt = f"""You are a helpful knowledge assistant. Answer based ONLY on the context provided.
    
    Context:
    {state['context']}
    
    Question:
    {state['question']}
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, base_url=BASE_URL, api_key=os.environ["OPENAI_API_KEY"])
    response = llm.invoke(prompt)
    return {"answer": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# ==========================================
# 💬 聊天界面 (UI)
# ==========================================
st.title("🤖 Knowledge Q&A Bot") # 🟢 改动点 2：页面主标题
st.caption("Ask specific questions about the knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    icon = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=icon):
        st.markdown(message["content"])

# 处理输入
if prompt := st.chat_input("Ask a question..."):
    # 1. 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 2. AI 消息
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🧠 AI is thinking..."):
            try:
                result = app.invoke({"question": prompt})
                full_response = result["answer"]
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                full_response = "Sorry, I encountered an error."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})