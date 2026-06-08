from openai import AzureOpenAI

from backend.app.config import Settings


class AzureOpenAIEmbeddingsClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_embedding_api_key_effective,
            api_version=settings.azure_openai_embedding_api_version_effective,
            azure_endpoint=settings.azure_openai_embedding_endpoint_base,
            timeout=settings.request_timeout_seconds,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts_with_usage(texts)["embeddings"]

    def embed_texts_with_usage(self, texts: list[str]) -> dict:
        response = self.client.embeddings.create(
            model=self.settings.azure_openai_embedding_deployment,
            input=texts,
        )
        # Preserve original ordering by index to avoid mismatch on bulk inserts.
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        total_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
        return {
            "embeddings": embeddings,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "model_name": self.settings.azure_openai_embedding_deployment,
        }

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
