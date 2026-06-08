# Interview Notes

## Why local Chroma instead of Azure AI Search

- Chroma is easy to run locally with no admin or cloud provisioning.
- It supports persistent vectors on disk for realistic local development.
- It allows fast experimentation before cloud hardening.

## How this maps to Azure AI Search in production

- Replace Chroma read/write methods with Azure AI Search index APIs.
- Keep chunking, metadata, and retrieval interfaces unchanged where possible.
- Add hybrid retrieval and RBAC once production permissions are available.

## Why LangSmith is used

- It provides trace-level visibility across retrieval, prompt, and LLM generation.
- It helps debug latency and quality issues by showing step-by-step execution.

## Why RAGAS is used

- It gives standardized RAG quality metrics.
- It helps measure improvements during prompt or retrieval changes.

## RAGAS vs LangSmith

- RAGAS measures answer quality (faithfulness, relevancy, context metrics).
- LangSmith traces runtime behavior and workflow observability.
- They are complementary: quality scoring plus execution tracing.

## App Insights vs Datadog vs LangSmith

- App Insights: application telemetry, errors, dependency calls, and Azure-native dashboards.
- Datadog: broader infra and application observability across multi-cloud systems.
- LangSmith: LLM application tracing and chain-level debugging.

## RAG vs agentic RAG

- RAG: retrieve context then generate answer in a single pipeline.
- Agentic RAG: agent decides tools/actions dynamically, often with planning loops.
- This project uses standard RAG for simplicity and reliability.

## Why this project does not need multi-agent setup

- Single-domain Q&A is well served by deterministic retrieve-then-generate.
- Multi-agent orchestration adds complexity, latency, and debugging overhead.

## Future deployment path to App Service or AKS

- Keep API contracts and modules unchanged.
- Add cloud storage + managed secrets + cloud vector DB.
- Containerize for AKS only when scale and team ops maturity require it.

## Common production issues and fixes

- Low retrieval quality -> improve chunking and corpus coverage.
- Hallucination risk -> tighten prompt grounding and output guardrails.
- High latency -> reduce top-k and context length, add caching.
- Cost growth -> track token usage and tune generation limits.
- Incident handling -> use runbooks, alerts, and replay traces.
