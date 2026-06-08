import logging


def log_latency(logger: logging.Logger, latency_ms: int, **extra) -> None:
    logger.info("latency_recorded", extra={"latency_ms": latency_ms, **extra})


def log_token_usage(logger: logging.Logger, estimated_tokens: int, model_name: str, **extra) -> None:
    logger.info(
        "token_usage_recorded",
        extra={"estimated_tokens": estimated_tokens, "model_name": model_name, **extra},
    )


def log_retrieval_score(logger: logging.Logger, retrieval_scores: list[float], **extra) -> None:
    logger.info(
        "retrieval_scores_recorded",
        extra={"retrieval_scores": retrieval_scores, "retrieved_doc_count": len(retrieval_scores), **extra},
    )


def log_guardrail_event(logger: logging.Logger, guardrail_status: str, **extra) -> None:
    logger.info("guardrail_event", extra={"guardrail_status": guardrail_status, **extra})


def log_error(logger: logging.Logger, error_type: str, message: str, **extra) -> None:
    logger.error(message, extra={"error_type": error_type, **extra})
