INJECTION_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "jailbreak",
    "developer message",
    "system message",
]


def check_input_guardrail(question: str) -> dict:
    lowered = question.lower()
    matches = [pattern for pattern in INJECTION_PATTERNS if pattern in lowered]

    if matches:
        return {
            "allowed": False,
            "status": "blocked_prompt_injection",
            "reason": f"Detected patterns: {', '.join(matches)}",
        }

    return {
        "allowed": True,
        "status": "passed",
        "reason": "No prompt injection patterns detected.",
    }
