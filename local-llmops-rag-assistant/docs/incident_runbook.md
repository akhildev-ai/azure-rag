# Incident Runbook

This runbook covers incidents simulated by `POST /simulate-incident`.

## openai_timeout

- Symptom: Chat endpoint latency spikes and then fails with timeout behavior.
- Detection: Error logs with timeout indicators and elevated latency_ms.
- Root cause: Slow Azure OpenAI responses, long prompts, or transient network issues.
- Resolution: Retry with backoff, reduce context size, lower max output length.

## vector_db_failure

- Symptom: Retrieval fails before generation.
- Detection: Error logs with vector store exception and retrieved_doc_count=0.
- Root cause: Chroma file lock, corrupted local DB files, or missing directory permissions.
- Resolution: Restart app, check `vector_db` path, rebuild index if needed.

## low_retrieval_score

- Symptom: Assistant frequently returns fallback answer.
- Detection: retrieval_scores values near zero and fallback response trend.
- Root cause: Poor chunking, sparse corpus, or mismatch between question style and docs.
- Resolution: Re-ingest documents with better chunking; add more representative data.

## prompt_injection

- Symptom: User question triggers guardrail block.
- Detection: Guardrail log event with status `blocked_prompt_injection`.
- Root cause: Malicious prompt pattern such as "ignore previous instructions".
- Resolution: Keep blocking; monitor repeated attempts by session/user id.

## high_token_usage

- Symptom: Cost and latency increase due to large prompt/answer size.
- Detection: estimated_tokens and latency_ms trends increase in logs.
- Root cause: Too many retrieved chunks or excessive output length.
- Resolution: Reduce context budget, trim prompt, enforce shorter answer length.
