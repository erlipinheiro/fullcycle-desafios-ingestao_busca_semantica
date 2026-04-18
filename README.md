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
OPENAI_API_KEY=<your_openai_api_key>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

GEMINI_API_KEY=<your_gemini_api_key>
GOOGLE_EMBEDDING_MODEL=models/embedding-001

DATABASE_URL=postgresql+psycopg://langchain:langchain@localhost:5432/langchain
PG_VECTOR_COLLECTION_NAME=document_chunks
PDF_PATH=../document.pdf
```

O sistema usa OpenAI por padrao quando `OPENAI_API_KEY` estiver definida. Se ela nao existir, usa Gemini quando `GEMINI_API_KEY` estiver definida. Se a OpenAI responder com erro de quota ou rate limit e `GEMINI_API_KEY` estiver configurada, a aplicacao faz fallback automatico para Gemini.

Descricao das variaveis:

- `OPENAI_API_KEY`: chave da OpenAI para embeddings e chat.
- `OPENAI_EMBEDDING_MODEL`: modelo de embedding usado com OpenAI. Valor padrao no exemplo: `text-embedding-3-small`.
- `GEMINI_API_KEY`: chave do Gemini para fallback de embeddings e chat.
- `GOOGLE_EMBEDDING_MODEL`: modelo de embedding usado com Gemini. Valor padrao no exemplo: `models/embedding-001`.
- `DATABASE_URL`: string de conexao do PostgreSQL com pgvector.
- `PG_VECTOR_COLLECTION_NAME`: nome da colecao usada para armazenar os chunks.
- `PDF_PATH`: caminho do PDF que sera ingerido.

Na pratica, a integracao atual do `langchain-google-genai` usa o identificador `models/embedding-001` para embeddings de texto no Gemini.

Se quiser manter os valores padrao do projeto, basta copiar o exemplo e substituir apenas as chaves de API:

```bash
cp .env.example .env
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

4. Ajuste `PDF_PATH` no arquivo `.env` para apontar para um PDF real que sera indexado.

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