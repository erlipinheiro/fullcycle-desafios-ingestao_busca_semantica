# Busca Semantica com LangChain, PGVector e CLI

Este projeto implementa um fluxo simples de RAG com duas etapas principais:

- ingestao de um PDF em chunks de 1000 caracteres com overlap de 150
- busca semantica com chat em linha de comando usando apenas o contexto recuperado

## Estrutura

```text
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── document.pdf
└── README.md
```

## Requisitos

- Python 3.11 ou superior
- Docker e Docker Compose
- Chave da OpenAI ou do Gemini

## Variaveis de ambiente

Crie um arquivo `.env` na raiz do projeto a partir do `.env.example`.

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
```

O sistema usa OpenAI por padrao quando `OPENAI_API_KEY` estiver definida. Se ela nao existir, usa Gemini quando `GEMINI_API_KEY` estiver definida. Se a OpenAI responder com erro de quota ou rate limit e `GEMINI_API_KEY` estiver configurada, a aplicacao faz fallback automatico para Gemini.

Na pratica, a integracao atual do `langchain-google-genai` usa o identificador `gemini-embedding-001` para embeddings de texto no Gemini.

Opcionalmente, voce pode sobrescrever a conexao do banco com:

```env
PGVECTOR_CONNECTION=postgresql+psycopg://langchain:langchain@localhost:5432/langchain
```

## Como executar

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Suba o PostgreSQL com pgvector:

```bash
docker compose up -d
```

4. Substitua o arquivo `document.pdf` por um PDF real com o conteudo que sera indexado.

5. Rode a ingestao:

```bash
python src/ingest.py
```

6. Rode a busca direta:

```bash
python src/search.py "Qual o faturamento da empresa?"
```

7. Rode o chat interativo:

```bash
python src/chat.py
```

## Comportamento esperado

- `src/ingest.py` carrega o PDF com `PyPDFLoader`, divide o conteudo com `RecursiveCharacterTextSplitter` e persiste os embeddings no PostgreSQL via `PGVector`.
- `src/search.py` executa `similarity_search_with_score(query, k=10)` e retorna os chunks mais relevantes.
- `src/chat.py` monta o prompt exigido, consulta a LLM e responde apenas com base no contexto recuperado.

Quando a informacao nao estiver explicitamente no contexto recuperado, a resposta esperada e:

```text
Não tenho informações necessárias para responder sua pergunta.
```

## Observacoes

- A ingestao usa `pre_delete_collection=True`, entao uma nova execucao recria a colecao para evitar duplicidade de chunks.
- Se nenhum contexto relevante for encontrado, o chat retorna a mensagem padrao sem consultar a LLM.
- Se a chave da OpenAI estiver sem saldo, configure `GEMINI_API_KEY` para manter ingestao, busca e chat operando sem mudar o codigo.
- Se OpenAI e Gemini estiverem sem quota ao mesmo tempo, a aplicacao interrompe a operacao com uma mensagem consolidada indicando que nao ha provider disponivel no momento.