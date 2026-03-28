from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "document.pdf"
COLLECTION_NAME = "document_chunks"
DEFAULT_DB_URL = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


def get_database_url() -> str:
    return os.getenv("PGVECTOR_CONNECTION", DEFAULT_DB_URL)


def get_provider_keys() -> dict[str, str]:
    provider_keys: dict[str, str] = {}
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if openai_api_key:
        provider_keys["openai"] = openai_api_key
    if gemini_api_key:
        provider_keys["gemini"] = gemini_api_key

    if not provider_keys:
        raise ValueError(
            "Defina OPENAI_API_KEY ou GEMINI_API_KEY no arquivo .env antes de rodar a ingestao."
        )

    return provider_keys


def get_embeddings(provider: str):
    provider_keys = get_provider_keys()

    if provider == "openai" and provider in provider_keys:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=provider_keys[provider])

    if provider == "gemini" and provider in provider_keys:
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
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
            "Atualize o saldo/cota da OpenAI ou Gemini, ou troque para chaves com uso disponivel."
        )

    details = "; ".join(f"{provider}: {error}" for provider, error in provider_errors.items())
    return f"Falha durante {operation}. Detalhes por provider: {details}"


def load_pdf_documents(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF nao encontrado em {pdf_path}. Substitua document.pdf por um arquivo valido."
        )

    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)


def ingest_with_provider(chunks, provider: str) -> None:
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(provider),
        collection_name=COLLECTION_NAME,
        connection=get_database_url(),
        use_jsonb=True,
        pre_delete_collection=True,
    )


def ingest_documents() -> int:
    load_environment()
    documents = load_pdf_documents(PDF_PATH)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado a partir do PDF informado.")

    provider_keys = get_provider_keys()
    provider_order = [provider for provider in ("openai", "gemini") if provider in provider_keys]
    last_error: Exception | None = None
    provider_errors: dict[str, Exception] = {}

    for index, provider in enumerate(provider_order):
        try:
            ingest_with_provider(chunks, provider)
            if index > 0:
                print(f"Ingestao concluida usando fallback de provider: {provider}.")
            return len(chunks)
        except Exception as exc:
            last_error = exc
            provider_errors[provider] = exc
            can_retry = provider == "openai" and "gemini" in provider_keys and is_quota_or_rate_limit_error(exc)
            if can_retry:
                print("OpenAI indisponivel por quota ou rate limit. Tentando Gemini para embeddings.")
                continue
            break

    if last_error and is_quota_or_rate_limit_error(last_error) and "gemini" not in provider_keys:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a ingestao")) from last_error

    if last_error:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a ingestao")) from last_error

    raise RuntimeError("Nenhum provider de embeddings disponivel para ingestao.")


def main() -> None:
    try:
        chunk_count = ingest_documents()
        print(f"Ingestao concluida com sucesso. {chunk_count} chunks foram armazenados.")
    except Exception as exc:
        print(f"Erro durante a ingestao: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
