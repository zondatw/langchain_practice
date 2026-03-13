"""
RAG Retrieval Level 2 Evaluator
比較 Dense vs Hybrid 搜尋在 magic-pack 專案的表現
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# ─────────────────────────────────────────────
# 測試案例定義
# ─────────────────────────────────────────────

@dataclass
class TestCase:
    question: str
    expected_source: str          # 最重要的來源檔名（不含路徑）
    expected_keywords: list[str]  # 出現在 retrieved content 中的關鍵字
    description: str = ""         # 這題在測什麼

TEST_CASES: list[TestCase] = [
    TestCase(
        question="如何安裝這個專案？",
        expected_source="README.md",
        expected_keywords=["cargo install", "magic-pack"],
        description="中文問安裝方式（關鍵字明確）"
    ),
    TestCase(
        question="How to install this project?",
        expected_source="README.md",
        expected_keywords=["cargo install", "magic-pack"],
        description="英文問安裝方式"
    ),
    TestCase(
        question="compress 函數的邏輯是什麼？",
        expected_source="main.rs",
        expected_keywords=["compress"],
        description="中文問特定函數（函數名為英文關鍵字）"
    ),
    TestCase(
        question="這個專案支援哪些壓縮格式？",
        expected_source="README.md",
        expected_keywords=["zip", "gzip"],
        description="中文問支援格式"
    ),
    TestCase(
        question="專案的核心功能是什麼？",
        expected_source="README.md",
        expected_keywords=["magic-pack"],
        description="中文問核心功能（語意模糊）"
    ),
    TestCase(
        question="interop tests check for what?",
        expected_source="interop.rs",
        expected_keywords=["PATH", "gzip", "zip"],
        description="英文問測試內容"
    ),
]

# ─────────────────────────────────────────────
# 評估結果
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    question: str
    description: str
    mode: str
    retrieved_sources: list[str]
    source_hit: bool
    keyword_hits: list[str]
    keyword_misses: list[str]
    latency_ms: float
    top1_source: str = ""

    @property
    def keyword_score(self) -> float:
        total = len(self.keyword_hits) + len(self.keyword_misses)
        return len(self.keyword_hits) / total if total > 0 else 0.0

    @property
    def overall_score(self) -> float:
        return (0.5 * int(self.source_hit)) + (0.5 * self.keyword_score)

# ─────────────────────────────────────────────
# 評估器
# ─────────────────────────────────────────────

class RAGEvaluator:
    def __init__(
        self,
        db_path: str = "./qdrant_db",
        collection_name: str = "magic_pack",
        k: int = 5,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _get_vectorstore(self, mode: str) -> QdrantVectorStore:
        retrieval_mode = (
            RetrievalMode.HYBRID if mode == "hybrid" else RetrievalMode.DENSE
        )
        sparse_embeddings = (
            FastEmbedSparse(model_name="Prithivida/Splade_PP_en_v1")
            if mode == "hybrid" else None
        )
        return QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            sparse_embedding=sparse_embeddings,
            path=self.db_path,
            collection_name=self.collection_name,
            retrieval_mode=retrieval_mode,
        )

    def _run_single(self, tc: TestCase, mode: str) -> EvalResult:
        vs = self._get_vectorstore(mode)
        retriever = vs.as_retriever(search_kwargs={"k": self.k})

        start = time.perf_counter()
        docs = retriever.invoke(tc.question)
        latency_ms = (time.perf_counter() - start) * 1000

        sources = [
            os.path.basename(d.metadata.get("source", "unknown")) for d in docs
        ]
        full_content = " ".join(d.page_content for d in docs)

        keyword_hits = [kw for kw in tc.expected_keywords if kw in full_content]
        keyword_misses = [kw for kw in tc.expected_keywords if kw not in full_content]

        return EvalResult(
            question=tc.question,
            description=tc.description,
            mode=mode,
            retrieved_sources=sources,
            top1_source=sources[0] if sources else "",
            source_hit=tc.expected_source in sources,
            keyword_hits=keyword_hits,
            keyword_misses=keyword_misses,
            latency_ms=latency_ms,
        )

    def run(
        self,
        modes: Optional[list[str]] = None,
        test_cases: Optional[list[TestCase]] = None,
    ) -> list[EvalResult]:
        if modes is None:
            modes = ["dense", "hybrid"]
        if test_cases is None:
            test_cases = TEST_CASES

        results = []
        for tc in test_cases:
            for mode in modes:
                result = self._run_single(tc, mode)
                results.append(result)
        return results

# ─────────────────────────────────────────────
# 報告輸出
# ─────────────────────────────────────────────

def print_report(results: list[EvalResult]):
    MODES = ["dense", "hybrid"]
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[96m"

    # ── 每題詳細結果 ──
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  詳細結果{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    questions = list(dict.fromkeys(r.question for r in results))
    for q in questions:
        q_results = {r.mode: r for r in results if r.question == q}
        ref = next(iter(q_results.values()))
        print(f"\n{CYAN}▶ {ref.description}{RESET}")
        print(f"  Q: {q}")

        for mode in MODES:
            if mode not in q_results:
                continue
            r = q_results[mode]
            src_icon = f"{GREEN}✅{RESET}" if r.source_hit else f"{RED}❌{RESET}"
            kw_color = GREEN if r.keyword_score == 1.0 else (YELLOW if r.keyword_score > 0 else RED)
            score_color = GREEN if r.overall_score >= 0.75 else (YELLOW if r.overall_score >= 0.5 else RED)

            print(f"  [{mode:6s}] "
                  f"source:{src_icon}  "
                  f"keywords:{kw_color}{len(r.keyword_hits)}/{len(r.keyword_hits)+len(r.keyword_misses)}{RESET}  "
                  f"score:{score_color}{r.overall_score:.0%}{RESET}  "
                  f"latency:{r.latency_ms:.0f}ms")

            if r.keyword_misses:
                print(f"           missing keywords: {RED}{r.keyword_misses}{RESET}")
            print(f"           retrieved: {r.retrieved_sources}")

    # ── 總分比較 ──
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  總分比較{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    summary: dict[str, dict] = {}
    for mode in MODES:
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        avg_score = sum(r.overall_score for r in mode_results) / len(mode_results)
        source_hit_rate = sum(r.source_hit for r in mode_results) / len(mode_results)
        avg_keyword = sum(r.keyword_score for r in mode_results) / len(mode_results)
        avg_latency = sum(r.latency_ms for r in mode_results) / len(mode_results)
        summary[mode] = {
            "avg_score": avg_score,
            "source_hit_rate": source_hit_rate,
            "avg_keyword": avg_keyword,
            "avg_latency": avg_latency,
        }
        color = GREEN if avg_score >= 0.75 else (YELLOW if avg_score >= 0.5 else RED)
        print(f"  {BOLD}{mode:8s}{RESET} "
              f"overall: {color}{avg_score:.0%}{RESET}  "
              f"source_hit: {source_hit_rate:.0%}  "
              f"keyword: {avg_keyword:.0%}  "
              f"avg_latency: {avg_latency:.0f}ms")

    # ── 結論 ──
    if len(summary) == 2:
        dense = summary.get("dense", {})
        hybrid = summary.get("hybrid", {})
        diff = hybrid.get("avg_score", 0) - dense.get("avg_score", 0)
        print(f"\n{BOLD}  結論{RESET}")
        if diff > 0.05:
            print(f"  {GREEN}➜ Hybrid 明顯優於 Dense (+{diff:.0%})，建議切換{RESET}")
        elif diff < -0.05:
            print(f"  {YELLOW}➜ Dense 反而較好，Hybrid 可能引入噪音{RESET}")
        else:
            print(f"  {YELLOW}➜ 兩者差異不大 ({diff:+.0%})，Dense 即可{RESET}")

    print(f"\n{BOLD}{'='*70}{RESET}\n")


def save_json(results: list[EvalResult], path: str = "eval_results.json"):
    data = [
        {
            "question": r.question,
            "description": r.description,
            "mode": r.mode,
            "source_hit": r.source_hit,
            "keyword_score": r.keyword_score,
            "overall_score": r.overall_score,
            "keyword_hits": r.keyword_hits,
            "keyword_misses": r.keyword_misses,
            "retrieved_sources": r.retrieved_sources,
            "latency_ms": round(r.latency_ms, 1),
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"結果已儲存至 {path}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = RAGEvaluator(
        db_path="./qdrant_db",
        collection_name="magic_pack",
        k=5,
    )

    print("🔍 開始評估 Dense vs Hybrid...")
    results = evaluator.run(modes=["dense", "hybrid"])
    print_report(results)
    save_json(results)