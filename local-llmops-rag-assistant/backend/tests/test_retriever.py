from backend.app.rag.retriever import Retriever


class DummyEmbeddings:
    def embed_query(self, _text):
        return [0.1, 0.2, 0.3]


class DummyStore:
    def similarity_search(self, query_embedding, top_k):
        assert query_embedding == [0.1, 0.2, 0.3]
        assert top_k == 5
        return [
            {"content": "doc1", "metadata": {"chunk_id": "1"}, "score": 0.7},
            {"content": "doc2", "metadata": {"chunk_id": "2"}, "score": 0.4},
        ]

    def lexical_search(self, query, top_k):
        assert query == "test question"
        assert top_k == 5
        return [
            {"content": "doc2", "metadata": {"chunk_id": "2"}, "score": 1.0},
            {"content": "doc3", "metadata": {"chunk_id": "3"}, "score": 0.5},
        ]


class DummySettings:
    top_k = 5
    enable_hybrid_retrieval = False
    hybrid_dense_weight = 0.6
    hybrid_lexical_weight = 0.4
    hybrid_rrf_k = 60
    enable_reranker = False
    reranker_model = "BAAI/bge-reranker-v2-m3"
    reranker_candidate_k = 20
    low_retrieval_score_threshold = 0.5


def test_retriever_returns_top_k_docs():
    retriever = Retriever(DummySettings(), DummyEmbeddings(), DummyStore())
    docs = retriever.retrieve("test question")
    assert len(docs) == 2


def test_retriever_low_score_detection():
    retriever = Retriever(DummySettings(), DummyEmbeddings(), DummyStore())
    assert retriever.is_low_score([{"score": 0.2}, {"score": 0.3}]) is True
    assert retriever.is_low_score([{"score": 0.8}]) is False


def test_retriever_hybrid_fuses_dense_and_lexical_results():
    settings = DummySettings()
    settings.enable_hybrid_retrieval = True
    retriever = Retriever(settings, DummyEmbeddings(), DummyStore())
    docs = retriever.retrieve("test question")
    contents = [doc["content"] for doc in docs]
    assert "doc3" in contents
