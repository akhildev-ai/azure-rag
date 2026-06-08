import re
import time
from collections import OrderedDict
from threading import Lock


class QueryResponseCache:
    def __init__(
        self,
        enabled: bool,
        ttl_seconds: int,
        max_entries: int,
        similarity_threshold: float,
        ngram_size: int,
        ngram_weight: float,
    ):
        self.enabled = enabled
        self.ttl_seconds = max(1, ttl_seconds)
        self.max_entries = max(1, max_entries)
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.ngram_size = max(2, ngram_size)
        self.ngram_weight = max(0.0, min(1.0, ngram_weight))
        self._lock = Lock()
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def lookup(self, question: str) -> dict | None:
        if not self.enabled:
            return None
        key = self._normalize(question)
        now = time.time()

        with self._lock:
            self._evict_expired(now)
            exact = self._store.get(key)
            if exact:
                created_at, payload = exact
                if now - created_at <= self.ttl_seconds:
                    self._store.move_to_end(key)
                    return payload
                self._store.pop(key, None)

            best_key = None
            best_payload = None
            best_score = 0.0
            for existing_key, (created_at, payload) in self._store.items():
                if now - created_at > self.ttl_seconds:
                    continue
                score = self._combined_similarity(key, existing_key)
                if score >= self.similarity_threshold and score > best_score:
                    best_key = existing_key
                    best_payload = payload
                    best_score = score

            if best_key and best_payload is not None:
                self._store.move_to_end(best_key)
                return best_payload

        return None

    def store(self, question: str, payload: dict) -> None:
        if not self.enabled:
            return
        key = self._normalize(question)
        now = time.time()

        with self._lock:
            self._evict_expired(now)
            self._store[key] = (now, payload)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_expired(self, now: float) -> None:
        expired_keys = [key for key, (created_at, _) in self._store.items() if now - created_at > self.ttl_seconds]
        for key in expired_keys:
            self._store.pop(key, None)

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = (text or "").strip().lower()
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    @staticmethod
    def _jaccard_similarity(left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        if union == 0:
            return 0.0
        return intersection / union

    def _combined_similarity(self, left: str, right: str) -> float:
        token_score = self._jaccard_similarity(left, right)
        ngram_score = self._ngram_jaccard_similarity(left, right, self.ngram_size)
        return (self.ngram_weight * ngram_score) + ((1.0 - self.ngram_weight) * token_score)

    @staticmethod
    def _ngram_jaccard_similarity(left: str, right: str, n: int) -> float:
        left_ngrams = QueryResponseCache._char_ngrams(left, n)
        right_ngrams = QueryResponseCache._char_ngrams(right, n)
        if not left_ngrams or not right_ngrams:
            return 0.0
        intersection = len(left_ngrams & right_ngrams)
        union = len(left_ngrams | right_ngrams)
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        compact = text.replace(" ", "")
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i : i + n] for i in range(0, len(compact) - n + 1)}
