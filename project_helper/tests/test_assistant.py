import importlib
import sys
import types
import unittest
from unittest import mock

from project_helper.settings import QdrantMode, QdrantSettings, ZhTwMcpSettings


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class AssistantModuleTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_patcher = mock.patch.dict(
            sys.modules,
            {
                "langchain_huggingface": _stub_module("langchain_huggingface", HuggingFaceEmbeddings=object),
                "langchain_ollama": _stub_module("langchain_ollama", ChatOllama=object),
                "langchain_chroma": _stub_module("langchain_chroma", Chroma=object),
                "langchain_qdrant": _stub_module(
                    "langchain_qdrant",
                    QdrantVectorStore=object,
                    RetrievalMode=types.SimpleNamespace(HYBRID="hybrid"),
                    FastEmbedSparse=object,
                ),
                "langchain_community.document_loaders": _stub_module(
                    "langchain_community.document_loaders",
                    DirectoryLoader=object,
                    TextLoader=object,
                ),
                "langchain_text_splitters": _stub_module(
                    "langchain_text_splitters",
                    Language=types.SimpleNamespace(RUST="rust"),
                    RecursiveCharacterTextSplitter=object,
                ),
                "langchain_core.prompts": _stub_module(
                    "langchain_core.prompts",
                    ChatPromptTemplate=object,
                ),
                "qdrant_client": _stub_module("qdrant_client", QdrantClient=object),
            },
            clear=False,
        )
        cls._module_patcher.start()
        sys.modules.pop("project_helper.assistant", None)
        cls.assistant_module = importlib.import_module("project_helper.assistant")
        cls.ZhTwMcpPostProcessor = cls.assistant_module.ZhTwMcpPostProcessor
        cls.RustProjectAssistant = cls.assistant_module.RustProjectAssistant

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("project_helper.assistant", None)
        cls._module_patcher.stop()

    def make_assistant(self, qdrant_settings: QdrantSettings | None = None):
        return self.RustProjectAssistant(
            project_path="/tmp/demo-project",
            qdrant_settings=qdrant_settings,
            embeddings=mock.sentinel.embeddings,
            model=mock.sentinel.model,
            zhtw_post_processor=mock.Mock(),
        )


class ZhTwMcpPostProcessorTest(AssistantModuleTestCase):
    def setUp(self):
        settings = ZhTwMcpSettings(enabled=False, command="zhtw-mcp")
        self.processor = self.ZhTwMcpPostProcessor(settings=settings)

    def test_optimize_returns_original_text_when_processor_unavailable(self):
        self.assertEqual(self.processor.optimize("original"), "original")

    def test_extract_text_prefers_structured_content(self):
        result = {
            "structuredContent": {"corrected_text": "fixed text"},
            "content": [{"type": "text", "text": '{"text":"ignored"}'}],
        }

        text = self.processor._extract_text(result, "original")

        self.assertEqual(text, "fixed text")

    def test_extract_text_supports_json_text_content(self):
        result = {
            "content": [{"type": "text", "text": '{"fixed_text":"fixed from content"}'}],
        }

        text = self.processor._extract_text(result, "original")

        self.assertEqual(text, "fixed from content")

    def test_extract_text_returns_original_when_payload_has_no_candidate(self):
        result = {
            "structuredContent": {"accepted": True},
            "content": [{"type": "text", "text": "plain text"}],
        }

        text = self.processor._extract_text(result, "original")

        self.assertEqual(text, "original")

    def test_cli_fallback_returns_original_when_stdout_empty(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("project_helper.assistant.subprocess.run", return_value=completed):
            text = self.processor._run_cli_fallback("original")

        self.assertEqual(text, "original")


class RustProjectAssistantVectorstoreTest(AssistantModuleTestCase):
    def test_get_vectorstore_returns_cached_instance(self):
        assistant = self.make_assistant()
        assistant._vs_cache["Chroma"] = mock.sentinel.cached

        result = assistant._get_vectorstore("Chroma")

        self.assertIs(result, mock.sentinel.cached)

    def test_get_vectorstore_builds_remote_qdrant_when_collection_missing(self):
        assistant = self.make_assistant(
            qdrant_settings=QdrantSettings(mode=QdrantMode.REMOTE, host="remote-host"),
        )

        with mock.patch.object(assistant, "_collection_exists", return_value=False), \
             mock.patch.object(assistant, "_build_index", return_value=mock.sentinel.built) as build_index:
            result = assistant._get_vectorstore("Qdrant")

        self.assertIs(result, mock.sentinel.built)
        build_index.assert_called_once_with("Qdrant")
        self.assertIs(assistant._vs_cache["Qdrant"], mock.sentinel.built)

    def test_get_vectorstore_loads_remote_qdrant_when_collection_exists(self):
        qdrant_settings = QdrantSettings(mode=QdrantMode.REMOTE, host="remote-host", port=7000)
        assistant = self.make_assistant(qdrant_settings=qdrant_settings)

        with mock.patch.object(assistant, "_collection_exists", return_value=True), \
             mock.patch.object(assistant, "_load_qdrant_vectorstore", return_value=mock.sentinel.remote_vs) as loader:
            result = assistant._get_vectorstore("Qdrant")

        self.assertIs(result, mock.sentinel.remote_vs)
        loader.assert_called_once_with(url=qdrant_settings.url)
        self.assertIs(assistant._vs_cache["Qdrant"], mock.sentinel.remote_vs)

    def test_get_vectorstore_builds_index_when_local_store_missing(self):
        assistant = self.make_assistant()

        with mock.patch.object(assistant, "_has_persisted_index", return_value=False), \
             mock.patch.object(assistant, "_build_index", return_value=mock.sentinel.built) as build_index:
            result = assistant._get_vectorstore("Chroma")

        self.assertIs(result, mock.sentinel.built)
        build_index.assert_called_once_with("Chroma")
        self.assertIs(assistant._vs_cache["Chroma"], mock.sentinel.built)

    def test_get_vectorstore_loads_existing_chroma_store(self):
        assistant = self.make_assistant()

        with mock.patch.object(assistant, "_has_persisted_index", return_value=True), \
             mock.patch.object(assistant, "_load_chroma_vectorstore", return_value=mock.sentinel.chroma_vs) as loader:
            result = assistant._get_vectorstore("Chroma")

        self.assertIs(result, mock.sentinel.chroma_vs)
        loader.assert_called_once_with("./chroma_db")
        self.assertIs(assistant._vs_cache["Chroma"], mock.sentinel.chroma_vs)

    def test_get_vectorstore_loads_existing_local_qdrant_store(self):
        assistant = self.make_assistant()

        with mock.patch.object(assistant, "_has_persisted_index", return_value=True), \
             mock.patch.object(assistant, "_load_qdrant_vectorstore", return_value=mock.sentinel.qdrant_vs) as loader:
            result = assistant._get_vectorstore("Qdrant")

        self.assertIs(result, mock.sentinel.qdrant_vs)
        loader.assert_called_once_with(path="./qdrant_db")
        self.assertIs(assistant._vs_cache["Qdrant"], mock.sentinel.qdrant_vs)


if __name__ == "__main__":
    unittest.main()
