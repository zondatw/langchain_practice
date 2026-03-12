import gradio as gr
import os
from assistant import RustProjectAssistant

def create_gr(assistant: RustProjectAssistant):
    def chat_response(message, history):
        # 處理 Gradio 的 history 格式並回傳答案
        return assistant.ask(message)

    # 這裡只處理佈局
    with gr.Blocks() as demo:
        gr.Markdown("# 🦀 Magic-Pack Rust 專案助手")
        
        with gr.Row():
            # 左側：聊天視窗
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_response,
                    examples=["這個專案的核心功能是什麼？", "請解釋 magic-pack 如何處理錯誤？"],
                )
            
            # 右側：專案資訊
            with gr.Column(scale=1):
                gr.Markdown("### 📂 已索引檔案")
                files = assistant.get_indexed_files()
                if files:
                    # 僅顯示檔名，增加可讀性
                    file_list_md = "\n".join([f"- `{os.path.basename(f)}`" for f in files])
                    gr.Markdown(file_list_md)
                else:
                    gr.Markdown("*目前無索引資料*")
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ 系統狀態")
                gr.Markdown("**模型:** `Llama3`\n\n**向量庫:** `ChromaDB`")
                
                # 預留按鈕
                re_scan_btn = gr.Button("重新掃描專案", variant="secondary")
                
    return demo

if __name__ == "__main__":
    PROJECT_PATH = "~/Repos/magic-pack"
    
    print("--- 正在初始化後端助手 ---")
    assistant = RustProjectAssistant(project_path=PROJECT_PATH)
    
    # 建立介面實例
    app = create_gr(assistant=assistant)
    
    print("--- 正在啟動 Gradio 伺服器 (http://localhost:7860) ---")
    
    # 移除 window_title，僅保留最基本的 server 設定與 theme
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme="soft" 
    )