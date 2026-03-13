import gradio as gr
import os
from assistant import RustProjectAssistant

def create_gr(assistant: RustProjectAssistant):
    def chat_response(message, history, db_type):
        return assistant.ask(message, db_type=db_type)

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
            
            # 右側：數據源狀態
            with gr.Column(scale=1):
                gr.Markdown("### 📂 數據源資訊")
                
                # 初始化檔案清單
                initial_files = assistant.get_indexed_files("Chroma")
                file_list_display = gr.Markdown(
                    "\n".join([f"- `{os.path.basename(f)}`" for f in initial_files]) 
                    if initial_files else "*掃描中...*"
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ 引擎狀態")
                status_md = gr.Markdown("**當前:** `Chroma` (預設)")
                
                # UI 聯動邏輯
                def update_ui_status(choice):
                    files = assistant.get_indexed_files(choice)
                    file_md = "\n".join([f"- `{os.path.basename(f)}`" for f in files]) if files else "*無資料*"
                    return f"**當前:** `{choice}`", file_md

                db_selector.change(
                    update_ui_status, 
                    inputs=[db_selector], 
                    outputs=[status_md, file_list_display]
                )
                
                gr.Button("重新掃描專案", variant="secondary")
                
    return demo

if __name__ == "__main__":
    # 配置你的專案路徑
    PROJECT_PATH = "~/Repos/magic-pack"
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = 7860
    
    print("--- 正在初始化後端助手 ---")
    assistant = RustProjectAssistant(project_path=PROJECT_PATH)
    
    app = create_gr(assistant=assistant)
    
    print(f"--- 正在啟動 Gradio 6.x 伺服器 (http://{SERVER_NAME}:{SERVER_PORT}) ---")
    app.launch(
        server_name=SERVER_NAME, 
        server_port=SERVER_PORT,
        theme="soft"
    )