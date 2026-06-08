from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    openai_configured: bool


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source_file: str
    page_number: int | None = None
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int
    trace_id: str | None = None
    trace_url: str | None = None
    guardrail_status: str


class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    question: str
    answer: str
    feedback: str = Field(..., pattern="^(thumbs_up|thumbs_down)$")
    trace_id: str | None = None


class UploadResponse(BaseModel):
    source_file: str
    chunk_count: int
    message: str


class IncidentRequest(BaseModel):
    incident_type: str = Field(
        ...,
        pattern="^(openai_timeout|vector_db_failure|low_retrieval_score|prompt_injection|high_token_usage)$",
    )
    question: str | None = None


class IncidentResponse(BaseModel):
    incident_type: str
    status: str
    details: dict[str, Any]
