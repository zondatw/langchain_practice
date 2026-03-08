import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self, persist_directory="./chroma_db", model_name="llama3"):
        self.persist_directory = persist_directory
        
        # 1. 初始化本地 Embedding (M3 Max 執行非常快)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. 初始化本地 LLM (Ollama)
        self.model = ChatOllama(model=model_name, temperature=0)
        
        # 3. 載入或建立向量資料庫
        self.vectorstore = self._init_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def _init_vectorstore(self):
        # 檢查本地是否已有資料夾且不為空
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"--- 載入現有資料庫: {self.persist_directory} ---")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("--- 建立新資料庫中... ---")
            # 初始種子數據
            initial_docs = [
                Document(page_content="公司的技術棧主要是 Rust, Golang 與 C++，並使用 Solana 進行 Web3 開發。", metadata={"source": "hr"}),
                Document(page_content="軟體工程師目前的專案是一個名為 magic-pack 的 Rust 壓縮工具。", metadata={"source": "project"}),
                Document(page_content="2026 年的團隊旅遊計畫去沖繩與鯨魚共游。", metadata={"source": "planning"})
            ]
            return Chroma.from_documents(
                documents=initial_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(self, texts: list[str], metadata: dict = None):
        """讓你可以隨時增加新文件到資料庫中"""
        new_docs = [Document(page_content=t, metadata=metadata or {}) for t in texts]
        self.vectorstore.add_documents(new_docs)
        print(f"--- 已新增 {len(texts)} 筆文件至資料庫 ---")

    def ask(self, question: str):
        # RAG 流程
        template = """僅根據以下提供的背景資訊來回答問題，若背景資訊不足以回答，請直接說明。
        <context>
        {context}
        </context>

        問題：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # 檢索
        relevant_docs = self.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # 鏈結並執行
        chain = prompt | self.model
        response = chain.invoke({"context": context, "question": question})
        return response.content

if __name__ == "__main__":
    # 初始化系統
    rag = RAGSystem()

    # 測試查詢
    print("\n[測試 1]")
    print(f"AI 回答：{rag.ask('我們公司用什麼語言開發 Web3？')}")

    # 模擬未來延續性：新增資訊
    rag.add_documents(["magic-pack 工具支援多執行緒壓縮，提升 50% 效能。"])
    
    print("\n[測試 2]")
    print(f"AI 回答：{rag.ask('magic-pack 的主要功能是什麼？')}")