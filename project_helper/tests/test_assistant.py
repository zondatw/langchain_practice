import unittest
import importlib
import sys
import types
from unittest import mock

from project_helper.settings import ZhTwMcpSettings


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub_module("langchain_huggingface", HuggingFaceEmbeddings=object)
_stub_module("langchain_ollama", ChatOllama=object)
_stub_module("langchain_chroma", Chroma=object)
_stub_module(
    "langchain_qdrant",
    QdrantVectorStore=object,
    RetrievalMode=types.SimpleNamespace(HYBRID="hybrid"),
    FastEmbedSparse=object,
)
_stub_module("langchain_community.document_loaders", DirectoryLoader=object, TextLoader=object)
_stub_module(
    "langchain_text_splitters",
    Language=types.SimpleNamespace(RUST="rust"),
    RecursiveCharacterTextSplitter=object,
)
_stub_module("langchain_core.prompts", ChatPromptTemplate=object)
_stub_module("qdrant_client", QdrantClient=object)

assistant_module = importlib.import_module("project_helper.assistant")
ZhTwMcpPostProcessor = assistant_module.ZhTwMcpPostProcessor


class ZhTwMcpPostProcessorTest(unittest.TestCase):
    def setUp(self):
        settings = ZhTwMcpSettings(enabled=False, command="zhtw-mcp")
        self.processor = ZhTwMcpPostProcessor(settings=settings)

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


if __name__ == "__main__":
    unittest.main()
