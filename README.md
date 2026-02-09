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
    User[用户提问] --> UI[Streamlit前端]
    UI --> Cache{Redis缓存?}
    Cache -- 命中 --> Return[返回缓存答案]
    Cache -- 未命中 --> Router[检索路由]
    
    subgraph HybridSearch [混合检索系统]
        Router --> Vector[向量搜索 ChromaDB]
        Router --> Keyword[关键词搜索 BM25]
        Vector --> Fuse[结果融合]
        Keyword --> Fuse
    end
    
    Fuse --> TopK[Top-K排序 k=6]
    TopK --> LLM[DeepSeek-V3模型]
    LLM --> UI
    LLM --> Update[更新Redis缓存]
