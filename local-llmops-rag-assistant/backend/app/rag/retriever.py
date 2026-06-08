from backend.app.config import Settings
from backend.app.rag.reranker import CrossEncoderReranker


class Retriever:
    def __init__(self, settings: Settings, embeddings_client, vector_store):
        self.settings = settings
        self.embeddings_client = embeddings_client
        self.vector_store = vector_store
        self.reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            enabled=settings.enable_reranker,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.settings.top_k
        candidate_k = max(top_k, self.settings.reranker_candidate_k) if self.settings.enable_reranker else top_k
        query_embedding = self.embeddings_client.embed_query(question)
        dense_candidates = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=candidate_k)
        candidates = dense_candidates

        if self.settings.enable_hybrid_retrieval:
            lexical_candidates = self.vector_store.lexical_search(query=question, top_k=candidate_k)
            candidates = self._rrf_fuse(dense_candidates=dense_candidates, lexical_candidates=lexical_candidates)

        if self.settings.enable_reranker:
            return self.reranker.rerank(question=question, docs=candidates, top_k=top_k)
        return candidates[:top_k]

    def is_low_score(self, retrieved_docs: list[dict]) -> bool:
        if not retrieved_docs:
            return True
        best_score = max(float(doc.get("score", 0.0)) for doc in retrieved_docs)
        return best_score < self.settings.low_retrieval_score_threshold

    @staticmethod
    def extract_scores(retrieved_docs: list[dict]) -> list[float]:
        return [float(doc.get("score", 0.0)) for doc in retrieved_docs]

    def _rrf_fuse(self, dense_candidates: list[dict], lexical_candidates: list[dict]) -> list[dict]:
        rrf_k = max(1, self.settings.hybrid_rrf_k)
        dense_weight = max(0.0, self.settings.hybrid_dense_weight)
        lexical_weight = max(0.0, self.settings.hybrid_lexical_weight)
        total_weight = max(dense_weight + lexical_weight, 1e-9)
        by_key: dict[str, dict] = {}

        def merge(candidates: list[dict], weight: float, branch: str) -> None:
            for rank, doc in enumerate(candidates, start=1):
                key = self._doc_key(doc)
                fused = by_key.setdefault(
                    key,
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": 0.0,
                    },
                )
                fused["score"] += weight / (rrf_k + rank)
                fused[f"{branch}_score"] = float(doc.get("score", 0.0))

        merge(dense_candidates, dense_weight, "dense")
        merge(lexical_candidates, lexical_weight, "lexical")

        # RRF scores are naturally tiny (e.g., ~1/(k+rank)); rescale to 0-1
        # so LOW_RETRIEVAL_SCORE_THRESHOLD remains interpretable.
        normalization = (rrf_k + 1) / total_weight
        for doc in by_key.values():
            doc["score"] = max(0.0, min(1.0, float(doc.get("score", 0.0) * normalization)))

        fused_docs = list(by_key.values())
        fused_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return fused_docs

    @staticmethod
    def _doc_key(doc: dict) -> str:
        metadata = doc.get("metadata", {}) or {}
        chunk_id = metadata.get("chunk_id")
        source = metadata.get("source")
        if chunk_id:
            return f"{source}:{chunk_id}" if source else str(chunk_id)
        return doc.get("content", "")[:200]
