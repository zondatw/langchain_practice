from .assistant import RustProjectAssistant
from .settings import load_settings


def Q_a_A(assistant: RustProjectAssistant, question: str):
    print(f"Q: {question}")
    print(f"AI Says: {assistant.ask(question)}")


def main() -> None:
    settings = load_settings()
    assistant = RustProjectAssistant(
        project_path=settings.project_path,
        runtime_settings=settings.runtime,
        qdrant_settings=settings.qdrant,
        zhtw_mcp_settings=settings.zhtw_mcp,
    )

    print("\n[AI 專案助手已就緒]")
    Q_a_A(assistant=assistant, question="magic-pack 的安裝步驟是什麼？")
    Q_a_A(assistant=assistant, question="magic-pack 怎麼實作的？")
    Q_a_A(assistant=assistant, question="magic-pack 作者有說未來想做什麼嗎？")


if __name__ == "__main__":
    main()
