import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in (
            "request_id",
            "session_id",
            "user_id",
            "latency_ms",
            "retrieved_doc_count",
            "retrieval_scores",
            "model_name",
            "estimated_tokens",
            "guardrail_status",
            "error_type",
            "trace_id",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("local_llmops_rag")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
