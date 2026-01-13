import os
import warnings
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔇 屏蔽烦人的警告信息，让输出更干净
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置区域
# ==========================================
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# 🤖 裁判 LLM：DeepSeek
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

# 🧠 向量模型：HuggingFace
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 2. 准备“考卷” (Golden Dataset)
# ==========================================
data_samples = {
    'question': [
        'What is the function of the cooling system?',
        'Explain the components of a diesel fuel system.',
        'Who is the president of the United States?'
    ],
    'answer': [
        'The cooling system circulates coolant through passages to cool specific components like the combustion chamber and valves.',
        'Based on the context provided, there is no information about diesel fuel systems. The text only discusses gasoline systems.',
        'The context does not provide information about the president.'
    ],
    'contexts': [
        ['The function of the cooling system is to circulate coolant through passages... to cool specific components.'], 
        ['Gasoline fuel system components include: fuel tank, lines, pump... (Diesel section missing)'],
        ['(Empty Context or Irrelevant Context)']
    ],
    'ground_truth': [
        'The cooling system circulates coolant to remove heat from the combustion chamber, valves, and other engine parts.',
        'The diesel fuel system components include the fuel tank, fuel lines, fuel pump, fuel filter, and injection system.',
        'The provided text does not contain political information.'
    ]
}

dataset = Dataset.from_dict(data_samples)

# ==========================================
# 3. 开始评估
# ==========================================
print("🚀 DeepSeek is acting as the Judge... (Evaluating 3 test cases)")
print("⏳ Please wait (~30 seconds)...")

try:
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=judge_llm,
        embeddings=embeddings_model
    )
    
    # ==========================================
    # 4. 输出成绩单 (修复版)
    # ==========================================
    print("\n=== 📊 RAG Evaluation Report ===")
    print(score)

    # 导出详细表格
    df = score.to_pandas()
    
    print("\n=== 📝 Detailed Scores (Full Table) ===")
    # 🔥 修复点：不再指定列名，直接打印前5列，防止报错
    pd.set_option('display.max_columns', None) # 显示所有列
    print(df)

    # 保存文件
    df.to_csv("rag_evaluation_report.csv", index=False)
    print("\n✅ Report saved to 'rag_evaluation_report.csv'")

except Exception as e:
    print(f"\n❌ Evaluation Failed: {e}")