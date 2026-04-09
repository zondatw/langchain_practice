import json
import logging
import select
import shlex
import shutil
import subprocess
import time

from .settings import ZhTwMcpSettings

logger = logging.getLogger("RustAssistant")


class ZhTwMcpPostProcessor:
    def __init__(self, settings: ZhTwMcpSettings | None = None):
        config = settings or ZhTwMcpSettings()
        self.enabled = config.enabled
        self.debug_enabled = config.debug_enabled
        self.command = shlex.split(config.command)
        self.timeout_seconds = config.timeout_seconds
        self.fix_mode = config.fix_mode
        self.profile = config.profile
        self.content_type = config.content_type
        self.output = config.output
        self.explain = config.explain
        self.max_errors = config.max_errors
        self.cli_fallback_enabled = config.cli_fallback_enabled
        self.available = self.enabled and bool(self.command) and shutil.which(self.command[0]) is not None

        if self.debug_enabled:
            logger.debug("ZHTW_MCP_DEBUG 已啟用，將輸出額外診斷資訊")

        if self.available:
            logger.info("zhtw-mcp 回答優化已啟用，command=%s", " ".join(self.command))
        elif self.enabled:
            logger.warning("找不到 zhtw-mcp 指令，將略過回答優化")

    def optimize(self, text: str) -> str:
        if not self.available or not text.strip():
            return text

        start = time.perf_counter()
        logger.info("zhtw-mcp optimize 開始，input_length=%s", len(text))
        self._debug_text_preview("zhtw-mcp optimize input", text)
        try:
            optimized = self._run_tool(text)
            if optimized and optimized != text:
                logger.info(
                    "zhtw-mcp 已套用回答優化，output_length=%s，elapsed=%.2fs",
                    len(optimized),
                    time.perf_counter() - start,
                )
                self._debug_compare(text, optimized, "mcp")
            else:
                logger.info(
                    "zhtw-mcp optimize 完成但內容未變更，elapsed=%.2fs",
                    time.perf_counter() - start,
                )
            return optimized or text
        except Exception as exc:
            logger.warning("zhtw-mcp MCP 模式失敗: %s", exc)
            if self.cli_fallback_enabled:
                try:
                    fallback_start = time.perf_counter()
                    logger.info("zhtw-mcp CLI fallback 開始")
                    optimized = self._run_cli_fallback(text)
                    if optimized and optimized != text:
                        logger.info(
                            "zhtw-mcp CLI fallback 成功，output_length=%s，elapsed=%.2fs",
                            len(optimized),
                            time.perf_counter() - fallback_start,
                        )
                        self._debug_compare(text, optimized, "cli")
                    else:
                        logger.info(
                            "zhtw-mcp CLI fallback 完成但內容未變更，elapsed=%.2fs",
                            time.perf_counter() - fallback_start,
                        )
                    return optimized or text
                except Exception as fallback_exc:
                    logger.warning("zhtw-mcp CLI fallback 失敗: %s", fallback_exc)
            logger.warning(
                "zhtw-mcp 回答優化失敗，改回原始回答，elapsed=%.2fs: %s",
                time.perf_counter() - start,
                exc,
            )
            return text

    def _run_tool(self, text: str) -> str:
        logger.info("啟動 zhtw-mcp subprocess，command=%s", " ".join(self.command))
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            logger.info("zhtw-mcp initialize request 準備送出")
            self._request(
                process,
                1,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "project-helper",
                        "version": "0.1.0",
                    },
                },
            )
            logger.info("zhtw-mcp initialize request 完成")
            logger.info("zhtw-mcp initialized notification 準備送出")
            self._notify(process, "notifications/initialized", {})
            logger.info("zhtw-mcp tools/call request 準備送出")
            response = self._request(
                process,
                2,
                "tools/call",
                {
                    "name": "zhtw",
                    "arguments": {
                        "text": text,
                        "fix_mode": self.fix_mode,
                        "max_errors": self.max_errors,
                        "profile": self.profile,
                        "content_type": self.content_type,
                        "explain": self.explain,
                        "output": self.output,
                    },
                },
            )
            logger.info("zhtw-mcp tools/call request 完成")
            return self._extract_text(response.get("result", {}), text)
        finally:
            self._close(process)

    def _run_cli_fallback(self, text: str) -> str:
        cli_command = [
            *self.command,
            "convert",
            "--content-type",
            self.content_type,
            "--",
        ]
        logger.info("執行 zhtw-mcp CLI fallback，command=%s", " ".join(cli_command))
        completed = subprocess.run(
            cli_command,
            input=text,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        logger.info(
            "zhtw-mcp CLI fallback 完成，returncode=%s，stdout_length=%s，stderr_length=%s",
            completed.returncode,
            len(completed.stdout),
            len(completed.stderr),
        )
        if completed.stderr.strip():
            logger.info("zhtw-mcp CLI fallback stderr: %s", completed.stderr.strip())
        if completed.returncode != 0:
            raise RuntimeError(f"CLI fallback failed with exit code {completed.returncode}")
        optimized = completed.stdout.strip() or text
        self._debug_text_preview("zhtw-mcp CLI fallback output", optimized)
        return optimized

    def _notify(self, process: subprocess.Popen, method: str, params: dict) -> None:
        self._send_message(process, {"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, process: subprocess.Popen, request_id: int, method: str, params: dict) -> dict:
        request_start = time.perf_counter()
        self._send_message(
            process,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        while True:
            message = self._read_message(process)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                logger.info(
                    "zhtw-mcp request 完成，method=%s，request_id=%s，elapsed=%.2fs",
                    method,
                    request_id,
                    time.perf_counter() - request_start,
                )
                return message
            if "id" in message and "method" in message:
                logger.info(
                    "zhtw-mcp 收到 server request，method=%s，request_id=%s",
                    message["method"],
                    message["id"],
                )
                self._send_message(
                    process,
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": f"Unsupported server request: {message['method']}",
                        },
                    },
                )

    def _send_message(self, process: subprocess.Popen, payload: dict) -> None:
        if process.stdin is None:
            raise RuntimeError("zhtw-mcp stdin is not available")
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        logger.debug(
            "zhtw-mcp send，method=%s，id=%s，bytes=%s",
            payload.get("method"),
            payload.get("id"),
            len(body),
        )
        process.stdin.write(header)
        process.stdin.write(body)
        process.stdin.flush()

    def _read_message(self, process: subprocess.Popen) -> dict:
        if process.stdout is None:
            raise RuntimeError("zhtw-mcp stdout is not available")

        headers = {}
        read_start = time.perf_counter()
        while True:
            line = self._readline_with_timeout(process, read_start)
            if not line:
                raise RuntimeError(self._build_process_error(process, "zhtw-mcp closed unexpectedly"))
            if line in (b"\r\n", b"\n"):
                break
            key, _, value = line.decode("ascii").partition(":")
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers["content-length"])
        logger.info("zhtw-mcp response headers 已收到，content_length=%s", content_length)
        body = self._read_exact_with_timeout(process, content_length, read_start)
        if len(body) != content_length:
            raise RuntimeError("Incomplete MCP response body")
        logger.info("zhtw-mcp response body 已收到，elapsed=%.2fs", time.perf_counter() - read_start)
        decoded_body = body.decode("utf-8")
        logger.debug("zhtw-mcp raw response body: %s", decoded_body)
        return json.loads(decoded_body)

    def _readline_with_timeout(self, process: subprocess.Popen, started_at: float) -> bytes:
        if process.stdout is None:
            raise RuntimeError("zhtw-mcp stdout is not available")
        self._wait_for_stdout(process, started_at, "等待 response header")
        return process.stdout.readline()

    def _read_exact_with_timeout(self, process: subprocess.Popen, size: int, started_at: float) -> bytes:
        if process.stdout is None:
            raise RuntimeError("zhtw-mcp stdout is not available")

        chunks = bytearray()
        while len(chunks) < size:
            self._wait_for_stdout(process, started_at, "等待 response body")
            chunk = process.stdout.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def _wait_for_stdout(self, process: subprocess.Popen, started_at: float, phase: str) -> None:
        if process.stdout is None:
            raise RuntimeError("zhtw-mcp stdout is not available")

        elapsed = time.perf_counter() - started_at
        remaining = self.timeout_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError(f"{phase} 超時，已等待 {elapsed:.2f}s")

        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if ready:
            return

        elapsed = time.perf_counter() - started_at
        raise TimeoutError(self._build_process_error(process, f"{phase} 超時，已等待 {elapsed:.2f}s"))

    def _build_process_error(self, process: subprocess.Popen, message: str) -> str:
        stderr_preview = ""
        if process.stderr is not None:
            try:
                ready, _, _ = select.select([process.stderr], [], [], 0)
                if ready:
                    stderr_preview = process.stderr.read1(4096).decode("utf-8", errors="ignore").strip()
            except Exception:
                stderr_preview = ""

        returncode = process.poll()
        details = [message, f"returncode={returncode}"]
        if stderr_preview:
            details.append(f"stderr={stderr_preview}")
        return "，".join(details)

    def _extract_text(self, result: dict, original_text: str) -> str:
        structured = result.get("structuredContent")
        candidate = self._extract_candidate(structured)
        if candidate:
            logger.debug("zhtw-mcp 從 structuredContent 擷取優化結果")
            self._debug_text_preview("zhtw-mcp structuredContent output", candidate)
            return candidate

        for item in result.get("content", []):
            if item.get("type") != "text":
                continue
            text = item.get("text", "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("zhtw-mcp content item 不是 JSON，略過")
                continue
            candidate = self._extract_candidate(parsed)
            if candidate:
                logger.debug("zhtw-mcp 從 content text JSON 擷取優化結果")
                self._debug_text_preview("zhtw-mcp content output", candidate)
                return candidate

        logger.debug("zhtw-mcp 回傳中找不到可用文字，保留原始回答")
        return original_text

    @staticmethod
    def _extract_candidate(payload) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("corrected_text", "fixed_text", "text", "output_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _close(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            logger.info("關閉 zhtw-mcp subprocess")
            process.terminate()
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.warning("zhtw-mcp subprocess terminate 超時，改用 kill()")
                process.kill()
                process.wait()

    def _debug_text_preview(self, label: str, text: str, max_chars: int = 240) -> None:
        if not self.debug_enabled:
            return
        preview = text[:max_chars].replace("\n", "\\n")
        logger.debug("%s: len=%s preview=%s", label, len(text), preview)

    def _debug_compare(self, before: str, after: str, source: str) -> None:
        if not self.debug_enabled:
            return
        logger.debug(
            "zhtw-mcp %s compare: before_len=%s after_len=%s changed=%s",
            source,
            len(before),
            len(after),
            before != after,
        )
        self._debug_text_preview(f"zhtw-mcp {source} before", before)
        self._debug_text_preview(f"zhtw-mcp {source} after", after)
