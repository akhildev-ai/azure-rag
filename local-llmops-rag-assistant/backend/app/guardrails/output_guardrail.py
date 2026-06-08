import re


SECRET_PATTERNS = [
    r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"secret\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"password\s*[:=]\s*\S+",
    r"token\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
]


def _is_grounded(answer: str, contexts: list[str]) -> bool:
    if not contexts:
        return True

    context_blob = " ".join(contexts).lower()
    answer_terms = [term for term in re.split(r"\W+", answer.lower()) if len(term) > 4]
    if not answer_terms:
        return True

    overlap = sum(1 for term in answer_terms if term in context_blob)
    return (overlap / max(1, len(answer_terms))) >= 0.1


def _contains_secret(answer: str) -> bool:
    lowered = answer.lower()
    return any(re.search(pattern, lowered) for pattern in SECRET_PATTERNS)


def check_output_guardrail(answer: str, retrieved_docs: list[dict], citations: list[dict]) -> dict:
    contexts = [doc.get("content", "") for doc in retrieved_docs]

    if _contains_secret(answer):
        return {
            "allowed": False,
            "status": "blocked_secret_exposure",
            "reason": "Potential secret-like string detected in output.",
        }

    if not _is_grounded(answer, contexts):
        return {
            "allowed": False,
            "status": "blocked_not_grounded",
            "reason": "Answer is weakly grounded in retrieved context.",
        }

    if contexts and not citations:
        return {
            "allowed": False,
            "status": "blocked_missing_citations",
            "reason": "Citations are required when context is used.",
        }

    return {
        "allowed": True,
        "status": "passed",
        "reason": "Output guardrail checks passed.",
    }
