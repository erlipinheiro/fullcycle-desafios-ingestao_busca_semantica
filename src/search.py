from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"


def _get_collection_name() -> str:
    return os.getenv("PG_VECTOR_COLLECTION_NAME") or "document_chunks"


def _get_openai_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _get_google_embedding_model() -> str:
    return os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/embedding-001"


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DB_URL)


def get_provider_keys() -> dict[str, str]:
    provider_keys: dict[str, str] = {}
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if openai_api_key:
        provider_keys["openai"] = openai_api_key
    if google_api_key:
        provider_keys["google"] = google_api_key

    if not provider_keys:
        raise ValueError(
            "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env antes de rodar a busca."
        )

    return provider_keys


def get_embeddings(provider: str):
    provider_keys = get_provider_keys()

    if provider == "openai" and provider in provider_keys:
        return OpenAIEmbeddings(model=_get_openai_embedding_model(), api_key=provider_keys[provider])

    if provider == "google" and provider in provider_keys:
        return GoogleGenerativeAIEmbeddings(
            model=_get_google_embedding_model(),
            google_api_key=provider_keys[provider],
        )

    raise ValueError(f"Provider de embeddings nao configurado: {provider}")


def is_quota_or_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "insufficient_quota",
        "quota",
        "rate limit",
        "rate_limit",
        "error code: 429",
        "429 you exceeded your current quota",
    )
    return any(marker in message for marker in retry_markers)


def build_provider_failure_message(provider_errors: dict[str, Exception], operation: str) -> str:
    if not provider_errors:
        return f"Nenhum provider de IA disponivel para {operation}."

    exhausted_providers = [
        provider for provider, error in provider_errors.items() if is_quota_or_rate_limit_error(error)
    ]
    if exhausted_providers and len(exhausted_providers) == len(provider_errors):
        provider_list = ", ".join(provider.title() for provider in exhausted_providers)
        return (
            f"Todos os providers configurados falharam por quota ou rate limit durante {operation}: {provider_list}. "
            "Atualize o saldo/cota da OpenAI ou Google, ou troque para chaves com uso disponivel."
        )

    details = "; ".join(f"{provider}: {error}" for provider, error in provider_errors.items())
    return f"Falha durante {operation}. Detalhes por provider: {details}"


def get_vector_store(provider: str) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(provider),
        collection_name=_get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )


def search_documents(query: str, k: int = 10):
    load_environment()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("A query informada esta vazia.")

    provider_keys = get_provider_keys()
    provider_order = [provider for provider in ("openai", "google") if provider in provider_keys]
    last_error: Exception | None = None
    provider_errors: dict[str, Exception] = {}

    for provider in provider_order:
        try:
            vector_store = get_vector_store(provider)
            return vector_store.similarity_search_with_score(clean_query, k=k)
        except Exception as exc:
            last_error = exc
            provider_errors[provider] = exc
            can_retry = provider == "openai" and "google" in provider_keys and is_quota_or_rate_limit_error(exc)
            if can_retry:
                print("OpenAI indisponivel por quota ou rate limit. Tentando Google para a busca semantica.")
                continue
            break

    if last_error and is_quota_or_rate_limit_error(last_error) and "google" not in provider_keys:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a busca semantica")) from last_error

    if last_error:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a busca semantica")) from last_error

    raise RuntimeError("Nenhum provider de embeddings disponivel para busca.")


def format_search_results(results) -> list[str]:
    formatted_results = []
    for index, (document, score) in enumerate(results, start=1):
        score_text = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
        formatted_results.append(
            f"[{index}] Score: {score_text}\n{document.page_content.strip()}"
        )
    return formatted_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca semantica em chunks armazenados no PGVector.")
    parser.add_argument("query", help="Pergunta para buscar os 10 chunks mais relevantes.")
    args = parser.parse_args()

    try:
        results = search_documents(args.query, k=10)
        if not results:
            print("Nenhum resultado encontrado.")
            return

        for item in format_search_results(results):
            print(item)
            print("-" * 80)
    except Exception as exc:
        print(f"Erro durante a busca: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
