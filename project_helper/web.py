import gradio as gr
from assistant import RustProjectAssistant

def create_gr(assistant: RustProjectAssistant):
    # --- Gradio 內部邏輯 ---
    def chat_response(message, history):
        """
        message: 使用者輸入的文字
        history: 對話歷史 (Gradio 會自動處理)
        """
        return assistant.ask(message)

    # 使用 Blocks 來包裹介面，這是目前設定主題最標準的做法
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🦀 Magic-Pack Rust 助手")
        gr.Markdown("輸入任何關於 magic-pack 專案的問題，我會幫你分析原始碼與文件。")
        
        gr.ChatInterface(
            fn=chat_response,
            examples=["這個專案的核心功能是什麼？", "請解釋 magic-pack 如何處理錯誤？", "如何安裝這個專案？"],
        )
    return demo

if __name__ == "__main__":
    # 配置路徑
    PROJECT_PATH = "~/Repos/magic-pack"
    
    print("--- 正在初始化後端助手 ---")
    assistant = RustProjectAssistant(project_path=PROJECT_PATH)
    
    # 建立介面
    demo = create_gr(assistant=assistant)
    
    print("--- 正在啟動 Gradio 伺服器 (http://localhost:7860) ---")
    demo.launch(server_name="0.0.0.0", server_port=7860)