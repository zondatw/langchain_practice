# project helper

- [Web](#web)
- [Test](#test)
- [docker](#docker)
- [k8s](#k8s)

## Web

`uv run web.py`

![](../readme_pictures/project_helper_web.png)  

token statistics  
![](../readme_pictures/token_statistics.png)  

## Shell

`uv run main.py`

```text
--- 初始化本地 Embedding 模型 (all-MiniLM-L6-v2) ---
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6370.19it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     | Details
------------------------+------------+--------
embeddings.position_ids | UNEXPECTED |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
--- 連線至本地 Ollama (Llama3) ---
--- 載入現有索引: ./chroma_db ---

[AI 專案助手已就緒]
Q: magic-pack 的安裝步驟是什麼？
AI Says: 根據README.md檔案的「Quick install」部分，magic-pack的安裝步驟是：

\`\`\`
cargo install magic-pack
\`\`\`

這個命令使用Cargo（Rust的包管理器）來安裝magic-pack。
Q: magic-pack 怎麼實作的？
AI Says: 🔮️ Based on the provided README.md file, it appears that `magic-pack` is a command-line tool written in Rust. The implementation details are not explicitly mentioned in the README file, but we can make some educated guesses based on the usage examples and options provided.

From the usage examples, we can see that `magic-pack` supports various compression formats (e.g., zip, tar, bz2, gz) and decompression operations. It also seems to support nested archives and auto-detection of file formats during decompression.

Given the simplicity of the tool's functionality, it is likely that `magic-pack` uses existing libraries or crates in Rust to handle the compression and decompression tasks. Some possible candidates include:

1. `zip` crate: For handling ZIP files.
2. `tar` crate: For handling TAR archives.
3. `flate2` crate: For handling compressed formats like GZ, BZ2, etc.

The tool's implementation might involve creating a simple CLI parser using a library like `clap` or `docopt`, and then delegating the compression/decompression tasks to the respective libraries or crates.

To confirm this hypothesis, we would need to inspect the source code of `magic-pack`. Unfortunately, the provided README file does not include the source code. If you have access to the source code, I'd be happy to help you analyze it and provide more specific insights into how `magic-pack` is implemented! 😊
Q: magic-pack 作者有說未來想做什麼嗎？
AI Says: 根據 README.md 文件，沒有明確提到 magic-pack 的作者有關未來計畫或目標的信息。

然而，我們可以從文件中找到一些可能的方向或開發方向：

1. 在 README.md 文件中，有一個「Reference」部分，列出了多個相關的資源和文件，這些資源可能會對 magic-pack 的開發或擴展產生影響。
2. 在命令選項中，有一個 `-l` 選項，可以設定壓縮或解壓縮的等級，這可能是作者想在未來增加更多的功能或選項的一個方向。

總之，沒有明確的信息，但可以從文件中找到一些可能的方向或開發方向。
```

## Test

`uv run test_prompt_injection.py --label "v4-prompt"`

![](../readme_pictures/pi_report.png)  

default report html path: `.test_result/injection_report.html`  

## docker

`docker compose up -d`

[grafana link](http://127.0.0.1:3000/d/rust-assistant/rust-assistant-monitor?orgId=1&from=now-6h&to=now&timezone=browser)  
![](../readme_pictures/docker_grafana.png)  

[prometheus link](http://localhost:9091/service-discovery)  
![](../readme_pictures/docker_prometheus.png)  


## k8s

`Need install k3d & helm first.`  

start:  
```shell
bash setup.sh

echo '127.0.0.1 qdrant.rust-assistant.local grafana.rust-assistant.local prometheus.rust-assistant.local' | sudo tee -a /etc/hosts"

kubectl port-forward svc/qdrant 6333:6333 -n qdrant &

QDRANT_MODE=remote QDRANT_HOST=localhost uv run web.py
```

end:  
```shell
bash teardown.sh
```

[grafana link](http://grafana.rust-assistant.local/d/rust-assistant/rust-assistant-monitor?orgId=1&from=now-6h&to=now&timezone=browser&refresh=10s)  
![](../readme_pictures/k8s_grafana.png) 

[prometheus link](http://prometheus.rust-assistant.local/targets)  
![](../readme_pictures/k8s_prometheus.png)  

[Qdrant link](http://qdrant.rust-assistant.local/dashboard#/collections)  
![](../readme_pictures/k8s_qdrant.png)  
