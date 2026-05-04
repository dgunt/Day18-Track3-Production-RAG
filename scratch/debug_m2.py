
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.m2_search import segment_vietnamese, BM25Search


CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

print("Segmenting query 'nghỉ phép':")
segmented_query = segment_vietnamese("nghỉ phép")
print(f"'{segmented_query}' -> tokens: {segmented_query.split()}")

print("\nSegmenting doc 0:")
segmented_doc = segment_vietnamese(CHUNKS[0]["text"])
print(f"'{segmented_doc}' -> tokens: {segmented_doc.split()}")

bm25 = BM25Search()
bm25.index(CHUNKS)
results = bm25.search("nghỉ phép", top_k=2)
print(f"\nSearch results for 'nghỉ phép': {results}")

results2 = bm25.search("nghỉ phép năm", top_k=2)
print(f"Search results for 'nghỉ phép năm': {results2}")
