from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from typing import Any
from uuid import uuid4

from backend.app.config import Settings


@dataclass
class TraceContext:
    trace_id: str
    trace_url: str | None


@dataclass
class SpanRecorder:
    outputs: dict[str, Any] | None = None

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


class LangSmithTracing:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: Any | None = self._build_client()

    def _build_client(self):
        if not self.settings.tracing_enabled:
            return None

        try:
            langsmith_module = importlib.import_module("langsmith")
            return langsmith_module.Client(api_key=self.settings.langsmith_api_key)
        except Exception:
            return None

    def create_trace(self) -> TraceContext:
        trace_id = str(uuid4())
        if not self.settings.tracing_enabled:
            return TraceContext(trace_id=trace_id, trace_url=None)

        trace_url = (
            f"https://smith.langchain.com/projects/{self.settings.langsmith_project}"
            f"?traceId={trace_id}"
        )
        return TraceContext(trace_id=trace_id, trace_url=trace_url)

    @contextmanager
    def span(self, name: str, trace_id: str):
        recorder = SpanRecorder()

        if not self.settings.tracing_enabled or self.client is None:
            yield recorder
            return

        run_id = str(uuid4())
        run_type = self._run_type_from_name(name)

        created = False
        try:
            self.client.create_run(
                id=run_id,
                name=name,
                run_type=run_type,
                project_name=self.settings.langsmith_project,
                inputs={"trace_id": trace_id, "span": name},
                start_time=datetime.now(timezone.utc),
                extra={"metadata": {"local_trace_id": trace_id}},
                tags=["local-rag", "fastapi"],
            )
            created = True
        except Exception:
            # Tracing should never block the API flow.
            created = False

        try:
            yield recorder
            if created:
                outputs = self._normalize_outputs_for_langsmith(run_type, recorder.outputs)
                self.client.update_run(
                    run_id,
                    outputs=outputs,
                    end_time=datetime.now(timezone.utc),
                )
        except Exception as exc:
            if created:
                try:
                    self.client.update_run(
                        run_id,
                        error=f"{type(exc).__name__}: {exc}",
                        end_time=datetime.now(timezone.utc),
                    )
                except Exception:
                    pass
            raise

    @staticmethod
    def _run_type_from_name(name: str) -> str:
        lowered = name.lower()
        if "embed" in lowered:
            return "embedding"
        if "retrieve" in lowered:
            return "retriever"
        if "generate" in lowered or "chat" in lowered:
            return "llm"
        if "upload" in lowered:
            return "tool"
        return "chain"

    @staticmethod
    def _normalize_outputs_for_langsmith(run_type: str, outputs: dict[str, Any] | None) -> dict[str, Any]:
        if not outputs:
            return {"status": "ok"}

        usage = outputs.get("usage_metadata") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        if run_type == "llm":
            normalized = {
                "status": outputs.get("status", "ok"),
                "model_name": outputs.get("model_name"),
                "llm_output": {
                    "token_usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                    "model_name": outputs.get("model_name"),
                },
            }
            if "answer_preview" in outputs:
                normalized["answer_preview"] = outputs["answer_preview"]
            return normalized

        if run_type == "embedding":
            return {
                "status": outputs.get("status", "ok"),
                "model_name": outputs.get("model_name"),
                "token_usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            }

        return outputs
