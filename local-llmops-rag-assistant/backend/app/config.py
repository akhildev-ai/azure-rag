from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(default="gpt-4o-mini", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-large", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    azure_openai_embedding_api_version: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_API_VERSION")
    azure_openai_embedding_api_key: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_API_KEY")
    azure_openai_embedding_endpoint: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_ENDPOINT")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="local-llmops-rag-assistant", alias="LANGSMITH_PROJECT")

    backend_host: str = Field(default="127.0.0.1", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    backend_base_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_BASE_URL")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    chroma_persist_directory: str = Field(default="vector_db", alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="rag_documents", alias="CHROMA_COLLECTION_NAME")
    uploads_directory: str = Field(default="data/uploads", alias="UPLOADS_DIRECTORY")
    log_file_path: str = Field(default="logs/app.log", alias="LOG_FILE_PATH")

    top_k: int = Field(default=5, alias="TOP_K")
    enable_hybrid_retrieval: bool = Field(default=False, alias="ENABLE_HYBRID_RETRIEVAL")
    hybrid_dense_weight: float = Field(default=0.6, alias="HYBRID_DENSE_WEIGHT")
    hybrid_lexical_weight: float = Field(default=0.4, alias="HYBRID_LEXICAL_WEIGHT")
    hybrid_rrf_k: int = Field(default=60, alias="HYBRID_RRF_K")
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_candidate_k: int = Field(default=20, alias="RERANKER_CANDIDATE_K")
    low_retrieval_score_threshold: float = Field(default=0.25, alias="LOW_RETRIEVAL_SCORE_THRESHOLD")
    max_context_chars: int = Field(default=12000, alias="MAX_CONTEXT_CHARS")
    max_chunks_in_prompt: int = Field(default=3, alias="MAX_CHUNKS_IN_PROMPT")
    max_chars_per_chunk: int = Field(default=900, alias="MAX_CHARS_PER_CHUNK")
    request_timeout_seconds: int = Field(default=45, alias="REQUEST_TIMEOUT_SECONDS")
    enable_query_cache: bool = Field(default=True, alias="ENABLE_QUERY_CACHE")
    query_cache_ttl_seconds: int = Field(default=1800, alias="QUERY_CACHE_TTL_SECONDS")
    query_cache_max_entries: int = Field(default=512, alias="QUERY_CACHE_MAX_ENTRIES")
    query_cache_similarity_threshold: float = Field(default=0.8, alias="QUERY_CACHE_SIMILARITY_THRESHOLD")
    query_cache_ngram_size: int = Field(default=3, alias="QUERY_CACHE_NGRAM_SIZE")
    query_cache_ngram_weight: float = Field(default=0.6, alias="QUERY_CACHE_NGRAM_WEIGHT")

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langsmith_tracing and self.langsmith_api_key and self.langsmith_project)

    @property
    def chroma_persist_path(self) -> Path:
        return self._resolve_project_path(self.chroma_persist_directory)

    @property
    def uploads_path(self) -> Path:
        return self._resolve_project_path(self.uploads_directory)

    @property
    def log_file(self) -> Path:
        return self._resolve_project_path(self.log_file_path)

    @property
    def azure_openai_chat_endpoint_base(self) -> str:
        return self._normalize_azure_endpoint(self.azure_openai_endpoint)

    @property
    def azure_openai_embedding_endpoint_base(self) -> str:
        endpoint = self.azure_openai_embedding_endpoint or self.azure_openai_endpoint
        return self._normalize_azure_endpoint(endpoint)

    @property
    def azure_openai_embedding_api_key_effective(self) -> str:
        return self.azure_openai_embedding_api_key or self.azure_openai_api_key

    @property
    def azure_openai_embedding_api_version_effective(self) -> str:
        return self.azure_openai_embedding_api_version or self.azure_openai_api_version

    @staticmethod
    def _normalize_azure_endpoint(endpoint: str) -> str:
        endpoint = (endpoint or "").strip()
        if not endpoint:
            return ""

        parsed = urlparse(endpoint)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

        # Keep original if parsing fails; upstream client will surface the error.
        return endpoint.rstrip("/")

    @staticmethod
    def _resolve_project_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    return settings
