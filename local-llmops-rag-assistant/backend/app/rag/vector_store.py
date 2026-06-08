from math import sqrt
import re

try:
    from chromadb import PersistentClient
except Exception:  # pragma: no cover - depends on local dependency installation
    PersistentClient = None

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency in some environments
    BM25Okapi = None

from backend.app.config import Settings


class ChromaVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._fallback_records: dict[str, dict] = {}

        if PersistentClient is None:
            self.client = None
            self.collection = None
            return

        self.client = PersistentClient(path=str(settings.chroma_persist_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if self.collection is None:
            for chunk, embedding in zip(chunks, embeddings):
                self._fallback_records[chunk["id"]] = {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding,
                }
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )

    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.collection is None:
            return self._fallback_similarity_search(query_embedding, top_k)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        normalized: list[dict] = []
        for content, metadata, distance in zip(docs, metas, distances):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            normalized.append(
                {
                    "content": content,
                    "metadata": metadata or {},
                    "score": score,
                }
            )

        return normalized

    def lexical_search(self, query: str, top_k: int) -> list[dict]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        docs = self._iter_documents()
        if not docs:
            return []

        if BM25Okapi is None:
            return self._token_overlap_search(query_terms, docs, top_k)

        tokenized_docs = [_tokenize(doc.get("content", "")) for doc in docs]
        if not any(tokenized_docs):
            return []

        bm25 = BM25Okapi(tokenized_docs)
        raw_scores = [float(value) for value in bm25.get_scores(query_terms)]
        max_score = max(raw_scores) if raw_scores else 0.0
        min_score = min(raw_scores) if raw_scores else 0.0
        denominator = (max_score - min_score) if max_score != min_score else 1.0

        candidates: list[dict] = []
        for item, raw_score in zip(docs, raw_scores):
            if raw_score <= 0.0:
                continue
            normalized = (raw_score - min_score) / denominator
            candidates.append(
                {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "score": max(0.0, min(1.0, float(normalized))),
                    "lexical_raw_score": raw_score,
                }
            )

        candidates.sort(key=lambda entry: entry["lexical_raw_score"], reverse=True)
        return candidates[:top_k]

    def is_ready(self) -> bool:
        try:
            if self.collection is None:
                return True
            _ = self.collection.count()
            return True
        except Exception:
            return False

    def _fallback_similarity_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        scored: list[dict] = []
        for item in self._fallback_records.values():
            score = _cosine_similarity(query_embedding, item["embedding"])
            scored.append(
                {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "score": max(0.0, min(1.0, score)),
                }
            )

        scored.sort(key=lambda entry: entry["score"], reverse=True)
        return scored[:top_k]

    def _iter_documents(self) -> list[dict]:
        if self.collection is None:
            return [
                {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                }
                for item in self._fallback_records.values()
            ]

        result = self.collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        entries: list[dict] = []
        for content, metadata in zip(docs, metas):
            entries.append(
                {
                    "content": content,
                    "metadata": metadata or {},
                }
            )
        return entries

    @staticmethod
    def _token_overlap_search(query_terms: list[str], docs: list[dict], top_k: int) -> list[dict]:
        query_set = set(query_terms)
        candidates: list[dict] = []
        for item in docs:
            content = item.get("content", "")
            doc_terms = set(_tokenize(content))
            if not doc_terms:
                continue

            common = len(query_set.intersection(doc_terms))
            if common == 0:
                continue

            score = common / max(len(query_set), 1)
            candidates.append(
                {
                    "content": content,
                    "metadata": item.get("metadata", {}),
                    "score": max(0.0, min(1.0, float(score))),
                }
            )

        candidates.sort(key=lambda entry: entry["score"], reverse=True)
        return candidates[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^\w]+", (text or "").lower()) if token]
