# Architecture

This project is a local production-style LLMOps RAG assistant using Python 3.11.

## Components

- Streamlit frontend for upload and chat UX.
- FastAPI backend for ingestion, retrieval, generation, feedback, and incident simulation APIs.
- Azure OpenAI for embeddings and chat completions.
- Chroma local persistent vector store in `vector_db`.
- Local JSON logs written to `logs/app.log`.
- Optional LangSmith tracing for request workflow visibility.
- RAGAS evaluation for quality metrics.

## Request Flow

```mermaid
flowchart TD
    A[User in Streamlit] --> B[FastAPI /chat]
    B --> C[Input Guardrail]
    C --> D[Retriever: Top 5 chunks from Chroma]
    D --> E{Low retrieval score?}
    E -- Yes --> F[Fallback answer]
    E -- No --> G[Prompt builder]
    G --> H[Azure OpenAI chat model]
    H --> I[Output Guardrail]
    I --> J[Response with citations]
    J --> A
```

## Upload Flow

```mermaid
flowchart TD
    A[Upload file] --> B[FastAPI /upload]
    B --> C[Save to data/uploads]
    C --> D[Load text from PDF/TXT/MD]
    D --> E[Chunk text]
    E --> F[Azure OpenAI embeddings]
    F --> G[Store vectors + metadata in Chroma]
```

## Local to Azure Production Mapping

- Local Chroma vector DB -> Azure AI Search vector index.
- Local file uploads -> Azure Blob Storage + ingestion jobs.
- Local logs file -> Azure Application Insights / Log Analytics.
- Local process manager -> Azure App Service or AKS.
- Local .env secrets -> Azure Key Vault + managed identity (when RBAC is available).
