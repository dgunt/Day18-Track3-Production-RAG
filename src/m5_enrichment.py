"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]
    if sentences:
        return " ".join(sentences[:3])

    return cleaned[:300].rstrip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    cleaned = " ".join(text.split())
    if not cleaned or n_questions <= 0 or not OPENAI_API_KEY:
        return []

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dua tren doan van, tao {n_questions} cau hoi ma doan van "
                        "co the tra loi. Tra ve moi cau hoi tren 1 dong, khong giai thich."
                    ),
                },
                {"role": "user", "content": cleaned[:4000]},
            ],
            max_tokens=min(80 * n_questions, 300),
            temperature=0,
        )
    except Exception:
        return []

    content = (resp.choices[0].message.content or "").strip()
    questions = []
    for line in content.splitlines():
        question = line.strip().lstrip("0123456789.-) ")
        if question:
            questions.append(question)
            if len(questions) >= n_questions:
                break
    return questions


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text:
        return ""

    title = " ".join(document_title.split())
    fallback_context = f"Excerpt from {title}." if title else ""

    if not OPENAI_API_KEY:
        return f"{fallback_context}\n\n{text}" if fallback_context else text

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write one short sentence describing where this passage fits "
                        "in the source document and what it is about. Return only that sentence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document: {title}\n\nPassage:\n{text[:4000]}",
                },
            ],
            max_tokens=80,
            temperature=0,
        )
    except Exception:
        return f"{fallback_context}\n\n{text}" if fallback_context else text

    context = (resp.choices[0].message.content or "").strip()
    if context:
        return f"{context}\n\n{text}"
    return f"{fallback_context}\n\n{text}" if fallback_context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    cleaned = " ".join(text.split())
    if not cleaned or not OPENAI_API_KEY:
        return {}

    try:
        import json
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Extract metadata from the passage. Return compact JSON with keys: '
                        '"topic" string, "entities" array of strings, "date_range" string, '
                        '"category" one of "policy", "hr", "it", "finance", "general", '
                        'and "language" one of "vi", "en", "other".'
                    ),
                },
                {"role": "user", "content": cleaned[:4000]},
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0,
        )
        metadata = json.loads((resp.choices[0].message.content or "").strip())
    except Exception:
        return {}

    if not isinstance(metadata, dict):
        return {}

    entities = metadata.get("entities", [])
    if isinstance(entities, str):
        metadata["entities"] = [entities]
    elif isinstance(entities, list):
        metadata["entities"] = [str(entity) for entity in entities if entity]
    else:
        metadata["entities"] = []

    return {
        key: metadata[key]
        for key in ("topic", "entities", "date_range", "category", "language")
        if key in metadata
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []
    method_set = set(methods)
    use_full = "full" in method_set
    use_summary = use_full or "summary" in method_set
    use_hyqa = use_full or "hyqa" in method_set
    use_contextual = use_full or "contextual" in method_set
    use_metadata = use_full or "metadata" in method_set
    method_name = "+".join(methods)

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata") or {}

        summary = summarize_chunk(text) if use_summary else ""
        questions = generate_hypothesis_questions(text) if use_hyqa else []
        enriched_text = (
            contextual_prepend(text, metadata.get("source", ""))
            if use_contextual
            else text
        )
        auto_meta = extract_metadata(text) if use_metadata else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text or text,
                summary=summary or "",
                hypothesis_questions=questions or [],
                auto_metadata={**metadata, **auto_meta},
                method=method_name,
            )
        )


    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
