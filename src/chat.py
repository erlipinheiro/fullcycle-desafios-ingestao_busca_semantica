from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from search import search_documents

BASE_DIR = Path(__file__).resolve().parent.parent
NO_CONTEXT_MESSAGE = "Não tenho informações necessárias para responder sua pergunta."
PROMPT_TEMPLATE = """CONTEXTO:
{context}

REGRAS:

* Responda somente com base no CONTEXTO.
* Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
* Nunca invente ou use conhecimento externo.
* Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"""


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


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
            "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env antes de rodar o chat."
        )

    return provider_keys


def get_llm(provider: str):
    provider_keys = get_provider_keys()

    if provider == "openai" and provider in provider_keys:
        return ChatOpenAI(model="gpt-5-nano", api_key=provider_keys[provider], temperature=0)

    if provider == "google" and provider in provider_keys:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=provider_keys[provider],
            temperature=0,
        )

    raise ValueError(f"Provider de LLM nao configurado: {provider}")


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


def build_context(question: str) -> str:
    results = search_documents(question, k=10)
    if not results:
        return ""

    return "\n\n".join(document.page_content.strip() for document, _score in results)


def answer_question(question: str) -> str:
    load_environment()
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("A pergunta informada esta vazia.")

    context = build_context(clean_question)
    if not context.strip():
        return NO_CONTEXT_MESSAGE

    prompt = PROMPT_TEMPLATE.format(context=context, question=clean_question)
    provider_keys = get_provider_keys()
    provider_order = [provider for provider in ("openai", "google") if provider in provider_keys]
    last_error: Exception | None = None
    provider_errors: dict[str, Exception] = {}

    for provider in provider_order:
        try:
            response = get_llm(provider).invoke(prompt)
            content = getattr(response, "content", "")

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(text)
                final_text = "\n".join(parts).strip()
            else:
                final_text = str(content).strip()

            return final_text or NO_CONTEXT_MESSAGE
        except Exception as exc:
            last_error = exc
            provider_errors[provider] = exc
            can_retry = provider == "openai" and "google" in provider_keys and is_quota_or_rate_limit_error(exc)
            if can_retry:
                print("OpenAI indisponivel por quota ou rate limit. Tentando Gemini para gerar a resposta.")
                continue
            break

    if last_error and is_quota_or_rate_limit_error(last_error) and "google" not in provider_keys:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a geracao da resposta")) from last_error

    if last_error:
        raise RuntimeError(build_provider_failure_message(provider_errors, "a geracao da resposta")) from last_error

    raise RuntimeError("Nenhum provider de LLM disponivel para gerar a resposta.")


def main() -> None:
    print("Chat RAG iniciado. Digite sua pergunta ou use 'sair' para encerrar.")

    while True:
        try:
            question = input("\nPergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            break

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        if not question:
            print("Digite uma pergunta valida.")
            continue

        try:
            print(f"\nResposta: {answer_question(question)}")
        except Exception as exc:
            print(f"Erro durante o chat: {exc}")


if __name__ == "__main__":
    main()
