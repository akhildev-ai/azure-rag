try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency in some environments
    CrossEncoder = None


class CrossEncoderReranker:
    def __init__(self, model_name: str, enabled: bool):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None

        if not enabled or CrossEncoder is None:
            return

        try:
            self._model = CrossEncoder(model_name)
        except Exception:
            # Keep retrieval functional even if model download/init fails.
            self._model = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._model is not None)

    def rerank(self, question: str, docs: list[dict], top_k: int) -> list[dict]:
        if not docs:
            return []
        if not self.active:
            return docs[:top_k]

        pairs = [(question, doc.get("content", "")) for doc in docs]
        scores = self._model.predict(pairs)

        rescored: list[dict] = []
        for doc, score in zip(docs, scores):
            updated = dict(doc)
            updated["reranker_score"] = float(score)
            rescored.append(updated)

        rescored.sort(key=lambda item: item.get("reranker_score", 0.0), reverse=True)
        return rescored[:top_k]
