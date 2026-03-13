import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

class RustProjectAssistant:
    def __init__(self, project_path="~/Repos/magic-pack"):
        self.project_path = os.path.abspath(os.path.expanduser(project_path))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = ChatOllama(model="llama3", temperature=0)
        self.db_paths = {
            "Chroma": "./chroma_db",
            "Qdrant": "./qdrant_db"
        }
        self.collection_name = "magic_pack"

    def _get_vectorstore(self, db_type):
        path = self.db_paths[db_type]
        if not os.path.exists(path) or not os.listdir(path):
            return self._build_index(db_type)

        if db_type == "Chroma":
            return Chroma(persist_directory=path, embedding_function=self.embeddings)
        else:
            return QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                path=path,
                collection_name=self.collection_name
            )

    def _build_index(self, db_type):
        print(f"--- 正在為 {db_type} 建立新索引 (3.13 相容模式) ---")
        all_docs = []

        exclude_dirs = ["/target/", "/.git/", "/.cargo/"]
        def load_and_filter(glob_pattern):
            loader = DirectoryLoader(
                self.project_path,
                glob=glob_pattern,
                loader_cls=TextLoader
            )
            raw_docs = loader.load()
            filtered_docs = [
                doc for doc in raw_docs 
                if not any(ex in doc.metadata.get('source', '') for ex in exclude_dirs)
            ]
            return filtered_docs

        print("--- [Step 1/2] 正在處理 Rust 檔案 (.rs) ---")
        rs_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RUST, chunk_size=1000, chunk_overlap=100
        )
        rs_docs = rs_splitter.split_documents(load_and_filter("**/*.rs"))
        all_docs.extend(rs_docs)

        sources = set(d.metadata['source'] for d in rs_docs)
        print(f"共載入 {len(rs_docs)} 個片段，來自 {len(sources)} 個檔案")
        for s in sorted(sources):
            print(s)

        print("--- [Step 2/2] 正在處理 Markdown 檔案 (.md) ---")
        md_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=80
        )
        all_docs.extend(md_splitter.split_documents(load_and_filter("**/*.md")))
        
        path = self.db_paths[db_type]
        if db_type == "Chroma":
            return Chroma.from_documents(all_docs, self.embeddings, persist_directory=path)
        else:
            return QdrantVectorStore.from_documents(
                all_docs, self.embeddings, path=path, collection_name=self.collection_name
            )

    def get_indexed_files(self, db_type="Chroma"):
        try:
            vs = self._get_vectorstore(db_type)
            sources = set()
            
            if db_type == "Chroma":
                data = vs.get()
                for metadata in data['metadatas']:
                    sources.add(metadata.get('source', 'unknown'))
            else:
                client = vs.client
                points, _ = client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    limit=5000
                )
                for p in points:
                    metadata = p.payload.get('metadata', p.payload)
                    if 'source' in metadata:
                        sources.add(metadata.get('source', 'unknown'))
            
            return sorted(list(set(s for s in sources if s)))
        except Exception as e:
            print(f"Error getting files: {e}")
            return []

    def _translate_to_english(self, question: str) -> str:
        response = self.model.invoke(
            f"Translate the following to English, output only the translation:\n{question}"
        )
        return response.content

    def ask(self, question: str, db_type="Chroma"):
        english_question = self._translate_to_english(question)
        print(f"Translated query: {english_question}")

        template = """
        你是一個專業的 Rust 開發助手。請根據以下專案背景（原始碼或文件）回答問題。
        若資訊不足以回答，請說明。回答時請引用檔案路徑並解釋邏輯。請用繁體中文回答

        <context>
        {context}
        </context>

        問題：{question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        vs = self._get_vectorstore(db_type)
        retriever = vs.as_retriever(search_kwargs={"k": 5})

        # 檢索與組合 Context
        context_docs = retriever.invoke(english_question)
        context_text = ""
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', '未知來源')
            context_text += f"--- 片段 {i+1} (來源: {source}) ---\n{doc.page_content}\n\n"
        print("---context_text---")
        print(context_text)
        chain = prompt | self.model
        response = chain.invoke({"context": context_text, "question": english_question})
        return response.content