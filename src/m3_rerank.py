"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._model_type: Optional[str] = None

    def _load_model(self) -> Optional[Any]:
        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._model_type = "flag"
            return self._model
        except Exception:
            pass

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._model_type = "crossencoder"
            return self._model
        except Exception:
            self._model = None
            self._model_type = None
            return None

    def _score_pairs(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        if hasattr(model, "compute_score"):
            scores = model.compute_score(pairs)
        elif hasattr(model, "predict"):
            scores = model.predict(pairs)
        else:
            raise RuntimeError("Unsupported reranker model interface")

        return [float(score) for score in scores]

    def _fallback_scores(self, query: str, documents: list[dict]) -> list[float]:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scores = []
        for doc in documents:
            text = doc.get("text", "").lower()
            overlap = sum(1 for token in query_tokens if token in text)
            score = float(doc.get("score", 0.0)) + overlap * 0.1
            scores.append(score)
        return scores

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]

        if model is not None:
            try:
                scores = self._score_pairs(model, pairs)
            except Exception:
                scores = self._fallback_scores(query, documents)
        else:
            scores = self._fallback_scores(query, documents)

        combined = [(score, doc) for score, doc in zip(scores, documents)]
        combined.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for i, (score, doc) in enumerate(combined[:top_k]):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            ))

        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            model = Ranker()
            passages = [{"text": d.get("text", "")} for d in documents]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            reranked: list[RerankResult] = []
            for i, res in enumerate(results[:top_k]):
                text = res.get("text") if isinstance(res, dict) else getattr(res, "text", "")
                score = res.get("score") if isinstance(res, dict) else getattr(res, "score", 0.0)
                original_doc = next((d for d in documents if d.get("text") == text), documents[0])
                reranked.append(RerankResult(
                    text=text,
                    original_score=float(original_doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=original_doc.get("metadata", {}),
                    rank=i + 1,
                ))
            return reranked
        except Exception:
            sorted_docs = sorted(documents, key=lambda d: float(d.get("score", 0.0)), reverse=True)
            return [
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i + 1,
                )
                for i, doc in enumerate(sorted_docs[:top_k])
            ]


def benchmark_reranker(reranker: Any, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "avg_ms": sum(times) / len(times) if times else 0.0,
        "min_ms": min(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
