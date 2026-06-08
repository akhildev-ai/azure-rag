from openai import AzureOpenAI

from backend.app.config import Settings
from backend.app.rag.prompt_templates import build_prompt


class AzureOpenAIChatGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_chat_endpoint_base,
            timeout=settings.request_timeout_seconds,
        )

    def generate(self, question: str, retrieved_docs: list[dict]) -> dict:
        prompt = build_prompt(
            question=question,
            retrieved_docs=retrieved_docs,
            max_context_chars=self.settings.max_context_chars,
            max_chunks_in_prompt=self.settings.max_chunks_in_prompt,
            max_chars_per_chunk=self.settings.max_chars_per_chunk,
        )

        response = self.client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": "Follow grounding rules and cite sources."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content or ""
        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
        estimated_tokens = total_tokens or max(1, len(answer) // 4)

        return {
            "answer": answer,
            "model_name": self.settings.azure_openai_chat_deployment,
            "estimated_tokens": estimated_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
