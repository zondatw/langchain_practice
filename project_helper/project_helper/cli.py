from .assistant import RustProjectAssistant
from .logging_utils import configure_logging
from .settings import load_settings


def ask_and_print(assistant: RustProjectAssistant, question: str) -> None:
    print(f"Q: {question}")
    print(f"AI Says: {assistant.ask(question)}")


def main() -> None:
    configure_logging()
    settings = load_settings()
    assistant = RustProjectAssistant(
        project_path=settings.project_path,
        runtime_settings=settings.runtime,
        qdrant_settings=settings.qdrant,
        zhtw_mcp_settings=settings.zhtw_mcp,
    )

    print("\n[AI 專案助手已就緒]")
    ask_and_print(assistant=assistant, question="magic-pack 的安裝步驟是什麼？")
    ask_and_print(assistant=assistant, question="magic-pack 怎麼實作的？")
    ask_and_print(assistant=assistant, question="magic-pack 作者有說未來想做什麼嗎？")
