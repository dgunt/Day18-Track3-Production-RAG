"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # TODO 1: Implement Vietnamese word segmentation
    from underthesea import word_tokenize
    # Why: BM25 needs word boundaries. "nghỉ phép" = 1 word, not 2.
    # Using format="text" to get a space-separated string of tokens.
    return word_tokenize(text, format="text")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # TODO 2: Implement BM25 indexing
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            # Segment Vietnamese text and split into tokens
            segmented_text = segment_vietnamese(chunk["text"])
            self.corpus_tokens.append(segmented_text.split())
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # TODO 3: Implement BM25 search
        if not self.bm25:
            return []

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score in descending order and get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [SearchResult(text=self.documents[i]["text"], score=scores[i], metadata=self.documents[i]["metadata"], method="bm25")
                for i in top_indices if scores[i] > 0] # Only return results with score > 0


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # TODO 4: Implement dense indexing
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Recreate collection to ensure a clean state and correct vector parameters
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        
        texts = [c["text"] for c in chunks]
        # Encode texts into vectors using the SentenceTransformer model
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        
        # Prepare PointStruct objects for upserting
        points = []
        for i, chunk in enumerate(chunks):
            payload = {**chunk["metadata"], "text": chunk["text"]} # Store original text and metadata in payload
            points.append(PointStruct(id=i, vector=vectors[i].tolist(), payload=payload))
            
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # TODO 5: Implement dense search
        query_vector = self._get_encoder().encode(query).tolist()
        hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
        
        return [SearchResult(text=hit.payload["text"], score=hit.score, metadata=hit.payload, method="dense") for hit in hits]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # TODO 6: Implement RRF
    rrf_scores = defaultdict(lambda: {"score": 0.0, "result": None})

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            # RRF formula: 1 / (k + rank + 1)
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            # Keep the original SearchResult object, or update if a higher score is found (optional, but good practice)
            if rrf_scores[result.text]["result"] is None or result.score > rrf_scores[result.text]["result"].score:
                rrf_scores[result.text]["result"] = result

    # Sort documents by their RRF score in descending order
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    # Convert back to SearchResult objects with method="hybrid"
    hybrid_results = []
    for item in sorted_results[:top_k]:
        original_result = item["result"]
        hybrid_results.append(
            SearchResult(
                text=original_result.text,
                score=item["score"], # Use the RRF score as the final score
                metadata=original_result.metadata,
                method="hybrid"
            )
        )
    return hybrid_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
