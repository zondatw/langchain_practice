import gradio as gr
import os
from assistant import RustProjectAssistant
from metrics import MetricsServer, instrument_assistant
from settings import load_settings

def create_gr(assistant: RustProjectAssistant):

    def chat_response(message, history, db_type):
        return assistant.ask(message, db_type=db_type)

    def get_token_display():
        usage = assistant.get_current_token_usages()
        return (
            f"| 類型 | Prompt | Completion | Total |\n"
            f"|------|--------|------------|-------|\n"
            f"| 翻譯 | {usage['translate']['prompt']} | {usage['translate']['completion']} | {usage['translate']['total']} |\n"
            f"| 回答 | {usage['ask']['prompt']} | {usage['ask']['completion']} | {usage['ask']['total']} |\n"
            f"| **合計** | **{usage['total']['prompt']}** | **{usage['total']['completion']}** | **{usage['total']['total']}** |"
        )

    def update_ui_status(choice):
        files = assistant.get_indexed_files(choice)
        file_md = "\n".join([f"- `{os.path.basename(f)}`" for f in files]) if files else "*無資料*"
        return f"**當前:** `{choice}`", file_md

    with gr.Blocks() as demo:
        gr.Markdown("# 🦀 Magic-Pack Rust 助手 (多引擎對比版)")

        with gr.Row():
            # 左側：聊天與引擎切換
            with gr.Column(scale=3):
                db_selector = gr.Radio(
                    choices=["Chroma", "Qdrant"],
                    value="Chroma",
                    label="選擇向量資料庫 (Vector DB)"
                )

                gr.ChatInterface(
                    fn=chat_response,
                    additional_inputs=[db_selector],
                    examples=[
                        ["如何安裝這個專案？", "Chroma"],
                        ["這個專案的核心功能是什麼？", "Chroma"],
                        ["如何處理壓縮邏輯？", "Qdrant"],
                        ["解釋 magic-pack 的錯誤處理機制", "Chroma"]
                    ],
                )

            # 右側：狀態面板
            with gr.Column(scale=1):
                gr.Markdown("### 📂 數據源資訊")
                initial_files = assistant.get_indexed_files("Chroma")
                file_list_display = gr.Markdown(
                    "\n".join([f"- `{os.path.basename(f)}`" for f in initial_files])
                    if initial_files else "*掃描中...*"
                )

                gr.Markdown("---")
                gr.Markdown("### ⚙️ 引擎狀態")
                status_md = gr.Markdown("**當前:** `Chroma` (預設)")

                db_selector.change(
                    update_ui_status,
                    inputs=[db_selector],
                    outputs=[status_md, file_list_display]
                )

                gr.Button("重新掃描專案", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### 📊 Token 使用量")
                token_display = gr.Markdown(get_token_display())
                refresh_btn = gr.Button("🔄 更新使用量", variant="secondary", size="sm")
                refresh_btn.click(fn=get_token_display, outputs=token_display)

                gr.Markdown("---")
                gr.Markdown(
                    "### 🔭 Metrics\n"
                    "Prometheus: `http://localhost:9091`  \n"
                    "Metrics: `http://localhost:9090/metrics`"
                )

    return demo


if __name__ == "__main__":
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = 7860
    METRICS_PORT = 9090
    settings = load_settings()

    print("--- 正在初始化後端助手 ---")
    assistant = RustProjectAssistant(
        project_path=settings.project_path,
        qdrant_settings=settings.qdrant,
        zhtw_mcp_settings=settings.zhtw_mcp,
    )

    print("--- 正在注入 Prometheus 監控 ---")
    instrument_assistant(assistant)
    MetricsServer.start(port=METRICS_PORT)

    app = create_gr(assistant=assistant)

    print(f"--- 正在啟動 Gradio 6.x 伺服器 (http://{SERVER_NAME}:{SERVER_PORT}) ---")
    app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        theme="soft"
    )
