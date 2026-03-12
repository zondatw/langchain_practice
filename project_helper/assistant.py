import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader

class RustProjectAssistant:
    def __init__(self, project_path="~/Repos/magic-pack", persist_directory="./chroma_db"):
        # 1. 處理路徑：展開 ~ 並轉為絕對路徑
        self.project_path = os.path.abspath(os.path.expanduser(project_path))
        self.persist_directory = persist_directory
        
        # 2. 初始化本地 Embedding (M3 Max 執行非常快)
        print("--- 初始化本地 Embedding 模型 (all-MiniLM-L6-v2) ---")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. 初始化本地 LLM (Ollama)
        print("--- 連線至本地 Ollama (Llama3) ---")
        self.model = ChatOllama(model="llama3", temperature=0)
        
        # 4. 建立或載入向量資料庫
        self.vectorstore = self._build_index()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def _build_index(self):
        self.project_path = os.path.abspath(os.path.expanduser(self.project_path))
        
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"--- 載入現有索引: {self.persist_directory} ---")
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        
        print(f"--- 開始掃描專案目錄 (免 Tree-Sitter 模式): {self.project_path} ---")
        all_docs = []

        # 1. 處理 Rust 原始碼 (改用 DirectoryLoader + TextLoader，繞過 tree-sitter)
        print("--- [Step 1/2] 正在處理 Rust 檔案 (.rs) ---")
        loader_rs = DirectoryLoader(
            self.project_path, 
            glob="**/*.rs", 
            loader_cls=TextLoader
        )
        # 依然可以用 RUST 的切分邏輯，它是基於換行與括號，不需編譯 tree-sitter
        rs_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RUST, chunk_size=1000, chunk_overlap=100
        )
        all_docs.extend(rs_splitter.split_documents(loader_rs.load()))

        # 2. 解析 Markdown 文件 (.md)
        print("--- [Step 2/2] 正在處理 Markdown 檔案 (.md) ---")
        loader_md = DirectoryLoader(
            self.project_path, 
            glob="**/*.md", 
            loader_cls=TextLoader
        )
        md_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=80
        )
        all_docs.extend(md_splitter.split_documents(loader_md.load()))

        if not all_docs:
            raise ValueError(f"在路徑 {self.project_path} 下找不到任何可索引的文件！")

        print(f"--- 成功切分出 {len(all_docs)} 個片段，正在建立向量資料庫... ---")
        
        # 建立並持久化到硬碟
        return Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def get_indexed_files(self):
        """獲取目前向量資料庫中所有來源檔案的清單 (去重複)"""
        if not self.vectorstore:
            return []
        # 從 metadata 中提取 source
        data = self.vectorstore.get()
        sources = set()
        for metadata in data['metadatas']:
            sources.add(metadata.get('source', 'unknown'))
        return sorted(list(sources))

    def ask(self, question: str):
        template = """你是一個專業的 Rust 開發助手。請根據以下專案背景（原始碼或文件）回答問題。
        若資訊不足以回答，請說明。回答時請引用檔案路徑並解釋邏輯。請用繁體中文回答

        <context>
        {context}
        </context>

        問題：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # 檢索與組合 Context
        context_docs = self.retriever.invoke(question)
        context_text = ""
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', '未知來源')
            context_text += f"--- 片段 {i+1} (來源: {source}) ---\n{doc.page_content}\n\n"
        
        # 執行 LLM 鏈結
        chain = prompt | self.model
        response = chain.invoke({"context": context_text, "question": question})
        
        return response.content