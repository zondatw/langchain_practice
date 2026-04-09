"""
Prometheus metrics interface for RustProjectAssistant
監控指標：token 使用量、response time、retrieval latency、error count

使用方式：
    from metrics import MetricsServer, instrument_assistant
    
    assistant = RustProjectAssistant(project_path="~/Repos/magic-pack")
    instrument_assistant(assistant)   # 注入監控
    
    MetricsServer.start(port=9090)    # 啟動 /metrics endpoint
"""

import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import wraps
from typing import Callable

# ─────────────────────────────────────────────
# 極簡 Prometheus metrics 實作（不依賴 prometheus_client）
# 如果環境有安裝 prometheus_client 可以直接換掉這段
# ─────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    )
    _USE_PROMETHEUS_CLIENT = True
except ImportError:
    _USE_PROMETHEUS_CLIENT = False


class _SimpleMetrics:
    """純 Python 的極簡 metrics 實作，格式符合 Prometheus text exposition"""

    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, labels: dict = None):
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def _key(self, name: str, labels: dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def exposition(self) -> str:
        """輸出符合 Prometheus text format 的 metrics"""
        lines = []
        with self._lock:
            for key, val in self._counters.items():
                name, labels = self._split_key(key)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels} {val}")

            for key, val in self._gauges.items():
                name, labels = self._split_key(key)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels} {val}")

            for key, vals in self._histograms.items():
                name, labels = self._split_key(key)
                count = len(vals)
                total = sum(vals)
                avg = total / count if count else 0
                # labels 插在 metric name 和 suffix 之間
                ls = labels if labels else ""
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{ls} {count}")
                lines.append(f"{name}_sum{ls} {total:.4f}")
                lines.append(f"{name}_avg{ls} {avg:.4f}")
                if vals:
                    sorted_vals = sorted(vals)
                    lines.append(f"{name}_p50{ls} {sorted_vals[int(count * 0.50)]:.4f}")
                    lines.append(f"{name}_p95{ls} {sorted_vals[min(int(count * 0.95), count-1)]:.4f}")
                    lines.append(f"{name}_p99{ls} {sorted_vals[min(int(count * 0.99), count-1)]:.4f}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _split_key(key: str):
        """把 'metric_name{label="val"}' 拆成 ('metric_name', '{label="val"}')"""
        if "{" in key:
            idx = key.index("{")
            return key[:idx], key[idx:]
        return key, ""


_metrics = _SimpleMetrics()
DEFAULT_METRICS_HOST = "0.0.0.0"
DEFAULT_METRICS_PORT = 9090

# ─────────────────────────────────────────────
# /metrics HTTP server
# ─────────────────────────────────────────────

class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = _metrics.exposition().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # 關閉 access log


class MetricsServer:
    _server: HTTPServer = None

    @classmethod
    def start(cls, port: int = DEFAULT_METRICS_PORT, host: str = DEFAULT_METRICS_HOST):
        if cls._server is not None:
            return
        cls._server = HTTPServer((host, port), _MetricsHandler)
        thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        thread.start()
        print(f"📊 Metrics server 啟動於 http://{host}:{port}/metrics")

    @classmethod
    def stop(cls):
        if cls._server:
            cls._server.shutdown()
            cls._server = None


# ─────────────────────────────────────────────
# Instrumentation：注入監控到 assistant
# ─────────────────────────────────────────────

def instrument_assistant(assistant) -> None:
    """
    monkey-patch RustProjectAssistant 的方法，注入 metrics 追蹤
    在 assistant 初始化後、開始使用前呼叫一次即可
    """
    _patch_translate(assistant)
    _patch_ask(assistant)


def _patch_translate(assistant) -> None:
    original = assistant._translate_to_english.__func__

    @wraps(original)
    def patched(self, question: str) -> str:
        start = time.perf_counter()
        try:
            result = original(self, question)
            _metrics.inc("rust_assistant_requests_total", labels={"operation": "translate", "status": "success"})
            return result
        except Exception as e:
            _metrics.inc("rust_assistant_requests_total", labels={"operation": "translate", "status": "error"})
            _metrics.inc("rust_assistant_errors_total", labels={"operation": "translate", "error": type(e).__name__})
            raise
        finally:
            elapsed = time.perf_counter() - start
            _metrics.observe("rust_assistant_latency_seconds", elapsed, labels={"operation": "translate"})

    assistant._translate_to_english = patched.__get__(assistant)


def _patch_ask(assistant) -> None:
    original = assistant.ask.__func__

    @wraps(original)
    def patched(self, question: str, db_type: str = "Chroma") -> str:
        start = time.perf_counter()
        try:
            result = original(self, question, db_type=db_type)

            # token 使用量（從 assistant 自己的 _token_usage 讀最新一次的差值）
            usage = self.get_current_token_usages()
            _metrics.set_gauge("rust_assistant_tokens_total_prompt",   usage["total"]["prompt"])
            _metrics.set_gauge("rust_assistant_tokens_total_completion", usage["total"]["completion"])
            _metrics.set_gauge("rust_assistant_tokens_total",           usage["total"]["total"])
            _metrics.set_gauge("rust_assistant_tokens_ask_prompt",     usage["ask"]["prompt"])
            _metrics.set_gauge("rust_assistant_tokens_ask_completion",  usage["ask"]["completion"])
            _metrics.set_gauge("rust_assistant_tokens_translate_prompt",     usage["translate"]["prompt"])
            _metrics.set_gauge("rust_assistant_tokens_translate_completion",  usage["translate"]["completion"])

            _metrics.inc("rust_assistant_requests_total", labels={"operation": "ask", "status": "success", "db": db_type})
            return result
        except Exception as e:
            _metrics.inc("rust_assistant_requests_total", labels={"operation": "ask", "status": "error", "db": db_type})
            _metrics.inc("rust_assistant_errors_total", labels={"operation": "ask", "error": type(e).__name__})
            raise
        finally:
            elapsed = time.perf_counter() - start
            _metrics.observe("rust_assistant_latency_seconds", elapsed, labels={"operation": "ask", "db": db_type})

    assistant.ask = patched.__get__(assistant)
