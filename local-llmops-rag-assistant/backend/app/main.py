import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import APIConnectionError, APIStatusError

from backend.app.config import get_settings
from backend.app.guardrails.input_guardrail import check_input_guardrail
from backend.app.guardrails.output_guardrail import check_output_guardrail
from backend.app.incidents.simulate_failures import simulate_incident
from backend.app.logging_config import configure_logging
from backend.app.observability.langsmith_tracing import LangSmithTracing
from backend.app.observability.metrics import (
    log_error,
    log_guardrail_event,
    log_latency,
    log_retrieval_score,
    log_token_usage,
)
from backend.app.rag.chunker import chunk_documents
from backend.app.rag.document_loader import load_document
from backend.app.rag.embeddings import AzureOpenAIEmbeddingsClient
from backend.app.rag.generator import AzureOpenAIChatGenerator
from backend.app.rag.query_cache import QueryResponseCache
from backend.app.rag.retriever import Retriever
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    HealthResponse,
    IncidentRequest,
    IncidentResponse,
    UploadResponse,
)

settings = get_settings()
logger = configure_logging(str(settings.log_file))
app = FastAPI(title="Local LLMOps RAG Assistant", version="1.0.0")


class Services:
    def __init__(self):
        self.embeddings_client = AzureOpenAIEmbeddingsClient(settings)
        self.vector_store = ChromaVectorStore(settings)
        self.retriever = Retriever(settings, self.embeddings_client, self.vector_store)
        self.generator = AzureOpenAIChatGenerator(settings)
        self.query_cache = QueryResponseCache(
            enabled=settings.enable_query_cache,
            ttl_seconds=settings.query_cache_ttl_seconds,
            max_entries=settings.query_cache_max_entries,
            similarity_threshold=settings.query_cache_similarity_threshold,
            ngram_size=settings.query_cache_ngram_size,
            ngram_weight=settings.query_cache_ngram_weight,
        )
        self.tracing = LangSmithTracing(settings)
        self._bootstrap_index_from_uploads()

    def _bootstrap_index_from_uploads(self) -> None:
        # If Chroma is unavailable, retrieval uses in-memory fallback and loses index on restart.
        # Re-index local uploaded files so chat still works after service restarts.
        candidates = sorted(
            [
                path
                for path in settings.uploads_path.glob("*")
                if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
            ]
        )
        if not candidates:
            return

        for path in candidates:
            try:
                docs = load_document(path)
                chunks = chunk_documents(docs)
                if not chunks:
                    continue
                embeddings = self.embeddings_client.embed_texts([chunk["content"] for chunk in chunks])
                self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            except Exception as exc:
                logger.warning(
                    "startup_reindex_skipped",
                    extra={"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"},
                )


services = Services()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        vector_store_ready=services.vector_store.is_ready(),
        openai_configured=bool(settings.azure_openai_api_key and settings.azure_openai_endpoint),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    request_id = str(uuid4())
    started = time.perf_counter()
    trace = services.tracing.create_trace()

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")

    target_path = settings.uploads_path / (file.filename or f"upload-{request_id}{suffix}")
    content = await file.read()
    target_path.write_bytes(content)

    try:
        with services.tracing.span("upload.load_and_chunk", trace.trace_id):
            docs = load_document(target_path)
            chunks = chunk_documents(docs)

        if not chunks:
            return UploadResponse(source_file=target_path.name, chunk_count=0, message="No text extracted.")

        with services.tracing.span("upload.embed_and_index", trace.trace_id) as span:
            embedding_result = services.embeddings_client.embed_texts_with_usage(
                [chunk["content"] for chunk in chunks]
            )
            embeddings = embedding_result["embeddings"]
            services.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            span.set_outputs(
                {
                    "status": "ok",
                    "model_name": embedding_result["model_name"],
                    "usage_metadata": {
                        "input_tokens": embedding_result["prompt_tokens"],
                        "output_tokens": 0,
                        "total_tokens": embedding_result["total_tokens"],
                    },
                }
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        log_latency(logger, latency_ms, request_id=request_id, trace_id=trace.trace_id)
        services.query_cache.clear()
        return UploadResponse(
            source_file=target_path.name,
            chunk_count=len(chunks),
            message="Document indexed successfully.",
        )
    except APIConnectionError as exc:
        log_error(
            logger,
            error_type=type(exc).__name__,
            message="upload_failed_connection",
            request_id=request_id,
            trace_id=trace.trace_id,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Upload failed: cannot connect to Azure OpenAI embeddings endpoint. "
                "Check AZURE_OPENAI_EMBEDDING_ENDPOINT (or AZURE_OPENAI_ENDPOINT), DNS/network access, and API version."
            ),
        ) from exc
    except APIStatusError as exc:
        log_error(
            logger,
            error_type=type(exc).__name__,
            message="upload_failed_api_status",
            request_id=request_id,
            trace_id=trace.trace_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Upload failed: Azure OpenAI returned status {exc.status_code}. Check deployment and API key.",
        ) from exc
    except Exception as exc:
        log_error(
            logger,
            error_type=type(exc).__name__,
            message="upload_failed",
            request_id=request_id,
            trace_id=trace.trace_id,
        )
        raise HTTPException(status_code=500, detail=f"Upload failed: {type(exc).__name__}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    request_id = str(uuid4())
    started = time.perf_counter()
    trace = services.tracing.create_trace()

    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        raise HTTPException(status_code=503, detail="Azure OpenAI credentials are not configured.")

    input_result = check_input_guardrail(request.question)
    if not input_result["allowed"]:
        log_guardrail_event(
            logger,
            input_result["status"],
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            trace_id=trace.trace_id,
        )
        raise HTTPException(status_code=400, detail=input_result["reason"])

    try:
        cached = services.query_cache.lookup(request.question)
        if cached is not None:
            latency_ms = int((time.perf_counter() - started) * 1000)
            log_latency(
                logger,
                latency_ms,
                request_id=request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                retrieved_doc_count=len(cached.get("citations", [])),
                model_name=cached.get("model_name", settings.azure_openai_chat_deployment),
                guardrail_status=cached.get("guardrail_status", "passed"),
                trace_id=trace.trace_id,
                cache_hit=True,
            )
            return ChatResponse(
                answer=cached["answer"],
                citations=[Citation(**item) for item in cached.get("citations", [])],
                latency_ms=latency_ms,
                trace_id=trace.trace_id,
                trace_url=trace.trace_url,
                guardrail_status=cached.get("guardrail_status", "passed"),
            )

        with services.tracing.span("chat.retrieve", trace.trace_id):
            retrieved_docs = services.retriever.retrieve(request.question, top_k=settings.top_k)

        retrieval_scores = services.retriever.extract_scores(retrieved_docs)
        log_retrieval_score(
            logger,
            retrieval_scores,
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            trace_id=trace.trace_id,
        )

        if services.retriever.is_low_score(retrieved_docs):
            latency_ms = int((time.perf_counter() - started) * 1000)
            fallback_answer = (
                "I do not have enough relevant context to answer confidently. "
                "Please upload more documents or rephrase the question."
            )
            log_latency(
                logger,
                latency_ms,
                request_id=request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                retrieved_doc_count=len(retrieved_docs),
                model_name=settings.azure_openai_chat_deployment,
                guardrail_status="passed",
                trace_id=trace.trace_id,
            )
            return ChatResponse(
                answer=fallback_answer,
                citations=[],
                latency_ms=latency_ms,
                trace_id=trace.trace_id,
                trace_url=trace.trace_url,
                guardrail_status="passed",
            )

        with services.tracing.span("chat.generate", trace.trace_id) as span:
            generation = services.generator.generate(request.question, retrieved_docs)
            span.set_outputs(
                {
                    "status": "ok",
                    "model_name": generation["model_name"],
                    "answer_preview": generation["answer"][:300],
                    "usage_metadata": {
                        "input_tokens": generation["prompt_tokens"],
                        "output_tokens": generation["completion_tokens"],
                        "total_tokens": generation["total_tokens"],
                    },
                }
            )

        citations = [
            Citation(
                source_file=doc.get("metadata", {}).get("source_file", "unknown"),
                page_number=doc.get("metadata", {}).get("page_number"),
                chunk_id=doc.get("metadata", {}).get("chunk_id", "unknown"),
                score=float(doc.get("score", 0.0)),
            )
            for doc in retrieved_docs
        ]

        output_result = check_output_guardrail(
            answer=generation["answer"],
            retrieved_docs=retrieved_docs,
            citations=[item.model_dump() for item in citations],
        )

        if not output_result["allowed"]:
            log_guardrail_event(
                logger,
                output_result["status"],
                request_id=request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                trace_id=trace.trace_id,
            )
            raise HTTPException(status_code=422, detail=output_result["reason"])

        latency_ms = int((time.perf_counter() - started) * 1000)
        log_latency(
            logger,
            latency_ms,
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            retrieved_doc_count=len(retrieved_docs),
            model_name=generation["model_name"],
            guardrail_status=output_result["status"],
            trace_id=trace.trace_id,
        )
        log_token_usage(
            logger,
            generation["estimated_tokens"],
            generation["model_name"],
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            trace_id=trace.trace_id,
        )

        services.query_cache.store(
            request.question,
            {
                "answer": generation["answer"],
                "citations": [item.model_dump() for item in citations],
                "guardrail_status": output_result["status"],
                "model_name": generation["model_name"],
            },
        )

        return ChatResponse(
            answer=generation["answer"],
            citations=citations,
            latency_ms=latency_ms,
            trace_id=trace.trace_id,
            trace_url=trace.trace_url,
            guardrail_status=output_result["status"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_error(
            logger,
            error_type=type(exc).__name__,
            message="chat_failed",
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            trace_id=trace.trace_id,
        )
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(exc).__name__}") from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    logger.info(
        "feedback_received",
        extra={
            "user_id": request.user_id,
            "session_id": request.session_id,
            "trace_id": request.trace_id,
            "guardrail_status": request.feedback,
        },
    )
    return {"status": "received"}


@app.post("/simulate-incident", response_model=IncidentResponse)
def simulate_incident_endpoint(request: IncidentRequest) -> IncidentResponse:
    details = simulate_incident(request.incident_type, request.question)
    logger.warning(
        "incident_simulated",
        extra={
            "error_type": request.incident_type,
            "guardrail_status": details.get("status", "unknown"),
        },
    )
    return IncidentResponse(**details)
