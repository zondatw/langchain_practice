import os
from langchain_huggingface import HuggingFaceEmbeddings
# 換成 Ollama
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 1. 準備數據 (保持不變)
docs = [
    Document(page_content="公司的技術棧主要是 Rust, Golang 與 C++，並使用 Solana 進行 Web3 開發。"),
    Document(page_content="後端工程師目前的專案是一個名為 magic-pack 的 Rust 壓縮工具。"),
    Document(page_content="2026 年的團隊旅遊計畫去沖繩與鯨鯊共游。")
]

# 2. 初始化本地 Embedding (保持不變)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. 初始化向量資料庫
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 4. 改用 Ollama 本地模型
# 建議使用 llama3 或 gemma2 (請確保你已經執行過 ollama pull llama3)
model = ChatOllama(model="llama3", temperature=0)

# 5. 設計 RAG 的提示詞模板
template = """僅根據以下提供的背景資訊來回答問題：
<context>
{context}
</context>

問題：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. 執行 RAG 流程
def ask_question(question):
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # 使用 LCEL 鏈結
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    
    # Ollama 的回傳物件結構與 OpenAI 略有不同，直接拿 content
    return response.content

if __name__ == "__main__":
    print("正在與本地模型連線...")
    res = ask_question("我們公司用什麼語言開發 Web3？")
    print(f"\nAI 回答：{res}")