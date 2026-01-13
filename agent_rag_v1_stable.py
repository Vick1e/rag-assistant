import os
import sys
import redis
from typing import List
from typing_extensions import TypedDict

# --- 1. Imports (兼容性修复版) ---
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_community.cache import RedisCache

# 🔥 修复 1：自动检测 LangChain 版本
try:
    from langchain.globals import set_llm_cache
except ImportError:
    import langchain
    def set_llm_cache(cache):
        langchain.llm_cache = cache
    print("⚠️ [System] Detected older LangChain version. Using compatibility mode.")

# ==========================================
# 👇 Configuration Area 👇
# ==========================================

# 1. DeepSeek API Key
os.environ["OPENAI_API_KEY"] = "sk-abf97993407943e698adda0bdeabddb8"

# 2. DeepSeek Base URL
BASE_URL = "https://api.deepseek.com"

# 3. Model Name
MODEL_NAME = "deepseek-chat"

# ==========================================
# 🚀 Enterprise Upgrade: Redis Caching Layer (万能修复版)
# ==========================================
print("🔌 [System] Connecting to Redis Cache...")
try:
    # 1. 先建立 Redis 连接
    client = redis.Redis(host="localhost", port=6379, db=0)
    
    # 2. 尝试多种参数写法，直到成功为止
    try:
        # 写法 A: 新版标准
        set_llm_cache(RedisCache(redis_client=client))
    except TypeError:
        try:
            # 写法 B: 你的版本提示的参数 (redis_)
            set_llm_cache(RedisCache(redis_=client))
        except TypeError:
            # 写法 C: 另一种旧版写法 (redis_url)
            set_llm_cache(RedisCache(redis_url="redis://localhost:6379/0"))
            
    print("✅ [Cache] Redis connected! Repeated queries will have near-zero latency.")
except Exception as e:
    print(f"⚠️ [Cache] Redis connection failed. Running without cache. Error: {e}")

# ==========================================

# --- 2. Data Processing (ETL Pipeline) ---
print("⚙️ [ETL] Loading and cleaning data...")

if not os.path.exists("manual_parsed.md"):
    print("❌ Error: 'manual_parsed.md' not found! Please check the file path.")
    sys.exit(1)

loader = UnstructuredMarkdownLoader("manual_parsed.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"✅ [ETL] Document split into {len(splits)} semantic chunks")

# --- 3. Vector Database (Vector Store) ---
print("💾 [DB] Loading local vector database...")

# 🔥 Local Embeddings (Privacy-first)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    collection_name="manual_rag",
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4. Define Graph State ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str

# --- 5. Define Nodes ---
def retrieve(state: GraphState):
    """Retrieval Node"""
    print(f"🔍 [Retriever] Retrieving info for: '{state['question']}'...")
    documents = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in documents])
    return {"context": context}

def generate(state: GraphState):
    """Generation Node"""
    print("🤖 [Generator] LLM is thinking (checking cache first)...")
    
    prompt = f"""You are a professional technical consultant. Answer the user's question based ONLY on the context provided below.
    
    If the answer is not in the context, simply say "I don't know based on the provided documents." Do not make up information.
    
    Context:
    {state['context']}
    
    Question:
    {state['question']}
    """
    
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        temperature=0, 
        base_url=BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- 6. Build Workflow (LangGraph) ---
print("🔗 [Graph] Compiling LangGraph workflow...")
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- 7. Execution ---
if __name__ == "__main__":
    question = "What are the course chapters listed in this manual? Please summarize them."
    
    print(f"\n🚀 Starting Task: {question}")
    
    try:
        result = app.invoke({"question": question})
        print("\n" + "="*30 + " FINAL ANSWER " + "="*30)
        print(result["answer"])
        print("="*74)
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
