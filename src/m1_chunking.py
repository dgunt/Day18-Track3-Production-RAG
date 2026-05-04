"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
from dataclasses import dataclass, field

import numpy as np
from numpy import dot
from numpy.linalg import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Tính cosine similarity giữa 2 vector."""
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(dot(a, b) / denom)


# Cache model instance để tránh load nhiều lần
_SENTENCE_MODEL = None


def _get_sentence_model():
    """
    Lazy-load SentenceTransformer để tránh load lúc import.
    Ưu tiên dùng cache local (local_files_only=True) để tránh crash network trên Windows.
    """
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL

    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = "all-MiniLM-L6-v2"
    _SENTENCE_MODEL = SentenceTransformer(model_name)

    return _SENTENCE_MODEL


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # Bước 1: Tách văn bản thành các câu — ưu tiên dùng underthesea cho tiếng Việt
    try:
        from underthesea import sent_tokenize as vi_sent_tokenize  # type: ignore
        sentences = [s.strip() for s in vi_sent_tokenize(text) if s.strip()]
    except ImportError:
        # Fallback: regex split nếu underthesea chưa cài
        raw_parts = re.split(r'(?<=[.!?。])\s+|\n\n', text)
        sentences = [s.strip() for s in raw_parts if s.strip()]

    if not sentences:
        return []

    # Chỉ 1 câu → trả ngay 1 chunk
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={**metadata, "chunk_index": 0, "strategy": "semantic"},
        )]

    # Bước 2: Encode các câu thành vector embedding
    model = _get_sentence_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Bước 3 & 4: Gom câu theo cosine similarity liên tiếp
    chunks: list[Chunk] = []
    current_group: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Tương đồng thấp → tách chunk mới
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])

    # Đừng quên nhóm cuối cùng
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}

    # Bước 1: Tách text thành Parent chunks bằng cách gom paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    current_text = ""

    for para in paragraphs:
        # Nếu thêm đoạn này vượt quá parent_size và hiện đã có nội dung → flush parent
        if len(current_text) + len(para) > parent_size and current_text:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            ))
            current_text = ""
        current_text += para + "\n\n"

    # Flush parent cuối
    if current_text.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))

    # Bước 2: Tách mỗi Parent thành các Children (sliding window theo char)
    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text
        start = 0
        while start < len(parent_text):
            end = start + child_size
            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                ))
            start = end  # Không overlap để đơn giản; có thể thêm overlap nếu cần

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # Bước 1: Tách text theo các tiêu đề markdown H1, H2, H3
    # Giữ lại delimiter (capture group) để biết đâu là header
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    # Bước 2: Ghép header với nội dung ngay sau nó
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Flush chunk cũ (nếu có nội dung)
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Flush chunk cuối cùng
    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
        ))

    # Nếu văn bản không có header nào → trả về 1 chunk với toàn bộ text
    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure"},
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def _compute_stats(chunks: list[Chunk]) -> dict:
    """Tính thống kê cơ bản cho danh sách chunks."""
    if not chunks:
        return {"num_chunks": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths)),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_hierarchical_children: list[Chunk] = []
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        # Chạy 4 chiến lược
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        _, children = chunk_hierarchical(text, metadata=meta)
        all_hierarchical_children.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic":        _compute_stats(all_basic),
        "semantic":     _compute_stats(all_semantic),
        "hierarchical": _compute_stats(all_hierarchical_children),
        "structure":    _compute_stats(all_structure),
    }

    # In bảng so sánh ra console
    print("\n" + "=" * 60)
    print(f"{'Strategy':<16} | {'Chunks':>6} | {'Avg Len':>7} | {'Min':>5} | {'Max':>5}")
    print("-" * 60)
    for name, stats in results.items():
        print(
            f"{name:<16} | {stats['num_chunks']:>6} | "
            f"{stats['avg_len']:>7} | {stats['min_len']:>5} | {stats['max_len']:>5}"
        )
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
