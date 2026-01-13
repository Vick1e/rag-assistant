# 🏗️ Knowledge Q&A Bot (DeepSeek + Hybrid Search)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--V3-blueviolet)
![Redis](https://img.shields.io/badge/Cache-Redis-red)

> **A production-ready RAG system designed for technical documentation. Features Hybrid Search (BM25 + Vector), Redis Caching, and quantitative evaluation.**

---

## ⚡ Key Features (核心亮点)

* **🧠 Hybrid Search (双路召回)**: Combines **Dense Vector Search** (ChromaDB) with **Sparse Keyword Search** (BM25) using `EnsembleRetriever` (k=6) to solve keyword mismatch issues.
* **🚀 Redis Caching**: Reduces latency for repeated queries from ~3s to **<10ms**.
* **📊 Quantitative Evaluation**: Validated using **Ragas framework** with a Faithfulness score of **0.83** and optimized Context Recall.
* **🛡️ Enterprise Grade**: Includes automatic fallback mechanisms and strict anti-hallucination prompts.

---

## 🏗️ Architecture (系统架构)

```mermaid
graph TD
    User[User Query] --> UI[Streamlit Frontend]
    UI --> Cache{Redis Cache Hit?}
    Cache -- Yes --> Return[Return Cached Answer <10ms>]
    Cache -- No --> Router[Retrieval Engine]
    
    subgraph "Hybrid Retrieval System"
        Router --> Vector[Vector Search (ChromaDB)]
        Router --> Keyword[Keyword Search (BM25)]
        Vector --> Fuse[Ensemble Fusion]
        Keyword --> Fuse
    end
    
    Fuse --> TopK[Top-K Chunks (k=6)]
    TopK --> LLM[DeepSeek-V3 API]
    LLM --> UI
    LLM --> Update[Update Redis Cache]

    
🧪 Evaluation Metrics (评估数据)
Faithfulness,0.83, High reliability; resists hallucination.
Context Recall, High,Optimized by tuning Top-K from 3 to 6.
