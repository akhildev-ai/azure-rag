import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def _normalize_azure_endpoint(url: str) -> str:
    endpoint = (url or "").strip().rstrip("/")
    marker = "/openai/"
    idx = endpoint.lower().find(marker)
    if idx != -1:
        endpoint = endpoint[:idx]
    return endpoint


def _build_azure_clients(base_dir: Path) -> tuple[AzureChatOpenAI, AzureOpenAIEmbeddings]:
    project_root = base_dir.parent
    load_dotenv(project_root / ".env")

    chat_endpoint = _normalize_azure_endpoint(
        os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    )
    embedding_endpoint = _normalize_azure_endpoint(
        os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    )
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    chat_api_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or chat_api_version

    if not all([chat_endpoint, embedding_endpoint, chat_deployment, embedding_deployment, chat_api_key, embedding_api_key]):
        raise RuntimeError("Missing Azure OpenAI settings for RAGAS evaluation. Check .env values.")

    llm = AzureChatOpenAI(
        azure_endpoint=chat_endpoint,
        api_key=chat_api_key,
        api_version=chat_api_version,
        deployment_name=chat_deployment,
        temperature=0.0,
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=embedding_endpoint,
        api_key=embedding_api_key,
        api_version=embedding_api_version,
        azure_deployment=embedding_deployment,
    )
    return llm, embeddings


def run_ragas_evaluation() -> dict:
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "eval_dataset.json"
    output_path = base_dir / "ragas_results.json"
    llm, embeddings = _build_azure_clients(base_dir)

    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = Dataset.from_list(records)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    score_dict = result.to_pandas().mean(numeric_only=True).to_dict()
    output_path.write_text(json.dumps(score_dict, indent=2), encoding="utf-8")
    return score_dict


if __name__ == "__main__":
    scores = run_ragas_evaluation()
    print(json.dumps(scores, indent=2))
