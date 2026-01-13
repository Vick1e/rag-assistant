import os
import sys
import redis
from typing import List, Dict
from typing_extensions import TypedDict

# --- 1. Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_community.cache import RedisCache

# 🔥 关键修复 1：混合检索的“安全导入” (Safe Import)
# 就算环境缺包，代码也不会崩，而是自动降级
HYBRID_SEARCH_AVAILABLE = False
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    import rank_bm25 # 检查依赖库
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ [System] Hybrid Search modules loaded successfully.")
except ImportError as e:
    print(f"⚠️ [System] Hybrid Search modules missing ({e}). Falling back to Standard Vector Search.")

# 🔥 关键修复 2：LangChain 版本自动适配
try:
    from langchain.globals import set_llm_cache
except ImportError:
    import langchain
    def set_llm_cache(cache):
        langchain.llm_cache = cache

# ==========================================
# 👇 Configuration Area
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-abf97993407943e698adda0bdeabddb8"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ==========================================
# 🚀 Redis Caching (万能连接版)
# ==========================================
print("🔌 [System] Connecting to Redis Cache...")
try:
    client = redis.Redis(host="localhost", port=6379, db=0)
    try:
        set_llm_cache(RedisCache(redis_client=client))
    except TypeError:
        set_llm_cache(RedisCache(redis_=client))  
    print("✅ [Cache] Redis connected! High-speed mode active.")
except Exception as e:
    print(f"⚠️ [Cache] Failed: {e}. Continuing without cache.")

# ==========================================
# ⚙️ ETL Pipeline
# ==========================================
print("⚙️ [ETL] Loading data...")
if not os.path.exists("manual_parsed.md"):
    print("❌ Error: 'manual_parsed.md' not found!")
    sys.exit(1)

loader = UnstructuredMarkdownLoader("manual_parsed.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"✅ [ETL] Split into {len(splits)} chunks")

# ==========================================
# 🏹 Retrieval System (自动切换逻辑)
# ==========================================
print("💾 [DB] Loading Vector Database...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    collection_name="manual_rag",
    persist_directory="./chroma_db"
)

# 基础检索器 (Vector Only)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

if HYBRID_SEARCH_AVAILABLE:
    # --- 启用混合检索 (Hybrid) ---
    print("🏹 [Hybrid] Building BM25 Keyword Index...")
    try:
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3
        
        print("⚖️ [Hybrid] Fusing Vector + Keyword search...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[base_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        retriever = ensemble_retriever
    except Exception as e:
        print(f"⚠️ [Hybrid] Error during index build: {e}. Reverting to Vector Search.")
        retriever = base_retriever
else:
    # --- 降级为普通检索 (Vector Only) ---
    print("ℹ️ [System] Running in Standard Vector Mode.")
    retriever = base_retriever

# ==========================================
# 🤖 Agent Workflow
# ==========================================
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

def retrieve(state: GraphState):
    print(f"🔍 [Retriever] Searching for: '{state['question']}'...")
    documents = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in documents])
    return {"context": context}

def generate(state: GraphState):
    print("🤖 [Generator] LLM thinking...")
    prompt = f"""You are a technical consultant. Answer based ONLY on the context.
    Context: {state['context']}
    Question: {state['question']}
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, base_url=BASE_URL)
    response = llm.invoke(prompt)
    return {"answer": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

if __name__ == "__main__":
    question = "What are the course chapters listed in this manual?"
    print(f"\n🚀 Starting Task: {question}")
    try:
        result = app.invoke({"question": question})
        print("\n=== FINAL ANSWER ===\n" + result["answer"] + "\n====================")
    except Exception as e:
        print(f"\n❌ Error: {e}")
