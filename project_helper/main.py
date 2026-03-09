import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

class ProjectAssistant:
    def __init__(self, project_path="./docs", persist_directory="./chroma_db"):
        self.project_path = project_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = ChatOllama(model="llama3", temperature=0)
        
        # 初始化或更新索引
        self.vectorstore = self._build_index()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def _build_index(self):
        # 判斷是否需要重新掃描
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("--- 載入現有專案索引 ---")
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        
        print(f"--- 正在掃描專案目錄: {self.project_path} ---")
        
        # 1. 載入文件 (這裡可以過濾副檔名，例如 .md 或 .rs)
        loader = DirectoryLoader(self.project_path, glob="**/*.md", loader_cls=TextLoader)
        raw_docs = loader.load()
        
        # 2. 切分文件 (對程式碼來說，這步非常重要)
        # chunk_size 代表每一段文字的大小，overlap 是段落間重疊的部分，能確保語義不被切斷
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_docs)
        
        print(f"--- 已切分出 {len(documents)} 個片段，開始建立向量索引 ---")
        
        # 3. 建立並持久化
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def ask(self, question: str):
        template = """你是一個專業的開發助手，請根據以下文件內容回答關於專案的問題：
        <context>
        {context}
        </context>
        問題：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        context_docs = self.retriever.invoke(question)
        context = "\n".join([d.page_content for d in context_docs])
        
        chain = prompt | self.model
        return chain.invoke({"context": context, "question": question}).content
    

def Q_a_A(assistant: ProjectAssistant, question: str):
    print(f"Q: {question}")
    print(f"AI Says: {assistant.ask(question)}")


if __name__ == "__main__":
    # 假設你的 magic-pack 專案文件放在 ./magic-pack/docs 下
    assistant = ProjectAssistant(project_path="~/Repos/magic-pack")
    
    print("\n[AI 專案助手已就緒]")
    Q_a_A(assistant=assistant, question="magic-pack 的安裝步驟是什麼？")
    Q_a_A(assistant=assistant, question="magic-pack 怎麼實作的？")