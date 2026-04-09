import os
import unittest
from unittest import mock

from project_helper import settings as settings_module
from project_helper.settings import QdrantMode, load_settings


class LoadSettingsTest(unittest.TestCase):
    def test_load_settings_uses_env_values(self):
        env = {
            "PROJECT_PATH": "/tmp/demo-project",
            "EMBEDDING_MODEL_NAME": "demo-embedding",
            "SPARSE_EMBEDDING_MODEL_NAME": "demo-sparse",
            "CHAT_MODEL_NAME": "demo-chat",
            "CHAT_TEMPERATURE": "0.7",
            "CHROMA_DB_PATH": "/tmp/chroma-demo",
            "QDRANT_DB_PATH": "/tmp/qdrant-demo",
            "RUST_CHUNK_SIZE": "111",
            "RUST_CHUNK_OVERLAP": "22",
            "MARKDOWN_CHUNK_SIZE": "333",
            "MARKDOWN_CHUNK_OVERLAP": "44",
            "RETRIEVER_K": "9",
            "QDRANT_SCROLL_LIMIT": "777",
            "QDRANT_MODE": "remote",
            "QDRANT_HOST": "qdrant.internal",
            "QDRANT_PORT": "7444",
            "QDRANT_COLLECTION": "demo_collection",
            "ZHTW_MCP_ENABLED": "false",
            "ZHTW_MCP_DEBUG": "true",
            "ZHTW_MCP_COMMAND": "/tmp/zhtw-mcp",
            "ZHTW_MCP_TIMEOUT_SECONDS": "12.5",
            "ZHTW_MCP_FIX_MODE": "lexical",
            "ZHTW_MCP_PROFILE": "custom",
            "ZHTW_MCP_CONTENT_TYPE": "text",
            "ZHTW_MCP_OUTPUT": "verbose",
            "ZHTW_MCP_EXPLAIN": "1",
            "ZHTW_MCP_MAX_ERRORS": "4",
            "ZHTW_MCP_CLI_FALLBACK_ENABLED": "0",
        }

        with mock.patch.object(settings_module, "load_dotenv", return_value=None):
            with mock.patch.dict(os.environ, env, clear=True):
                settings = load_settings()

        self.assertEqual(settings.project_path, "/tmp/demo-project")
        self.assertEqual(settings.runtime.embedding_model_name, "demo-embedding")
        self.assertEqual(settings.runtime.sparse_embedding_model_name, "demo-sparse")
        self.assertEqual(settings.runtime.chat_model_name, "demo-chat")
        self.assertEqual(settings.runtime.chat_temperature, 0.7)
        self.assertEqual(settings.runtime.chroma_db_path, "/tmp/chroma-demo")
        self.assertEqual(settings.runtime.qdrant_db_path, "/tmp/qdrant-demo")
        self.assertEqual(settings.runtime.rust_chunk_size, 111)
        self.assertEqual(settings.runtime.rust_chunk_overlap, 22)
        self.assertEqual(settings.runtime.markdown_chunk_size, 333)
        self.assertEqual(settings.runtime.markdown_chunk_overlap, 44)
        self.assertEqual(settings.runtime.retriever_k, 9)
        self.assertEqual(settings.runtime.qdrant_scroll_limit, 777)
        self.assertEqual(settings.qdrant.mode, QdrantMode.REMOTE)
        self.assertTrue(settings.qdrant.is_remote)
        self.assertEqual(settings.qdrant.url, "http://qdrant.internal:7444")
        self.assertEqual(settings.qdrant.collection_name, "demo_collection")
        self.assertFalse(settings.zhtw_mcp.enabled)
        self.assertTrue(settings.zhtw_mcp.debug_enabled)
        self.assertEqual(settings.zhtw_mcp.command, "/tmp/zhtw-mcp")
        self.assertEqual(settings.zhtw_mcp.timeout_seconds, 12.5)
        self.assertEqual(settings.zhtw_mcp.fix_mode, "lexical")
        self.assertEqual(settings.zhtw_mcp.profile, "custom")
        self.assertEqual(settings.zhtw_mcp.content_type, "text")
        self.assertEqual(settings.zhtw_mcp.output, "verbose")
        self.assertTrue(settings.zhtw_mcp.explain)
        self.assertEqual(settings.zhtw_mcp.max_errors, 4)
        self.assertFalse(settings.zhtw_mcp.cli_fallback_enabled)

    def test_load_settings_uses_defaults_when_env_missing(self):
        with mock.patch.object(settings_module, "load_dotenv", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_settings()

        self.assertEqual(settings.project_path, "~/Repos/magic-pack")
        self.assertEqual(settings.qdrant.mode, QdrantMode.LOCAL)
        self.assertFalse(settings.qdrant.is_remote)
        self.assertEqual(settings.qdrant.url, "http://localhost:6333")
        self.assertEqual(settings.runtime.sparse_embedding_model_name, "Prithivida/Splade_PP_en_v1")
        self.assertEqual(settings.runtime.chroma_db_path, "./chroma_db")
        self.assertEqual(settings.runtime.qdrant_db_path, "./qdrant_db")
        self.assertEqual(settings.runtime.retriever_k, 5)
        self.assertTrue(settings.zhtw_mcp.enabled)
        self.assertTrue(settings.zhtw_mcp.cli_fallback_enabled)


if __name__ == "__main__":
    unittest.main()
