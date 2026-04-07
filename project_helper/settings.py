import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ZhTwMcpSettings:
    enabled: bool = True
    debug_enabled: bool = False
    command: str = "zhtw-mcp"
    timeout_seconds: float = 10.0
    fix_mode: str = "lexical_safe"
    profile: str = "default"
    content_type: str = "markdown"
    output: str = "compact"
    explain: bool = False
    max_errors: int = 0
    cli_fallback_enabled: bool = True


@dataclass(frozen=True)
class QdrantSettings:
    mode: str = "local"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "magic_pack"


@dataclass(frozen=True)
class AssistantSettings:
    project_path: str = "~/Repos/magic-pack"
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    zhtw_mcp: ZhTwMcpSettings = field(default_factory=ZhTwMcpSettings)


def load_settings() -> AssistantSettings:
    if load_dotenv is not None:
        load_dotenv()

    return AssistantSettings(
        project_path=os.environ.get("PROJECT_PATH", "~/Repos/magic-pack"),
        qdrant=QdrantSettings(
            mode=os.environ.get("QDRANT_MODE", "local"),
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            collection_name=os.environ.get("QDRANT_COLLECTION", "magic_pack"),
        ),
        zhtw_mcp=ZhTwMcpSettings(
            enabled=_env_flag("ZHTW_MCP_ENABLED", True),
            debug_enabled=_env_flag("ZHTW_MCP_DEBUG", False),
            command=os.environ.get("ZHTW_MCP_COMMAND", "zhtw-mcp"),
            timeout_seconds=float(os.environ.get("ZHTW_MCP_TIMEOUT_SECONDS", "10")),
            fix_mode=os.environ.get("ZHTW_MCP_FIX_MODE", "lexical_safe"),
            profile=os.environ.get("ZHTW_MCP_PROFILE", "default"),
            content_type=os.environ.get("ZHTW_MCP_CONTENT_TYPE", "markdown"),
            output=os.environ.get("ZHTW_MCP_OUTPUT", "compact"),
            explain=_env_flag("ZHTW_MCP_EXPLAIN", False),
            max_errors=int(os.environ.get("ZHTW_MCP_MAX_ERRORS", "0")),
            cli_fallback_enabled=_env_flag("ZHTW_MCP_CLI_FALLBACK_ENABLED", True),
        ),
    )
