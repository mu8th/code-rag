# code-rag

A local, dependency-light **RAG code assistant**. Point it at a Python codebase,
ask a question, and it answers with a grounded, source-cited response, built
entirely on your machine.

The pipeline is four stages:

```
ingest            embed             retrieve            synthesize
──────────  ────────────────  ────────────────  ────────────────────
walk .py files   batch-embed    cosine top-k over   local LLM answers
chunk by symbol  (Ollama nomic) the embedded index  from the passages,
   │                  │                 │              citing file/symbol/lines
   └──────────────────┴─────────────────┴──────────────►  RagResult
```

- **Ingest**: standard-library `ast` splits each file into one chunk per
  top-level function/class (plus the module preamble), so retrieval lands on the
  right *symbol*.
- **Embed**: Ollama's `/api/embed` (nomic-embed-text) turns chunks into 768-dim
  vectors, batched over HTTP.
- **Retrieve**: a pure-numpy cosine index returns the top-k most relevant
  passages.
- **Synthesize**: an OpenAI-compatible local chat endpoint (LM Studio) answers
  using only the retrieved context, citing the sources it relied on.

## Why local?

Everything runs against the developer's own model endpoints. Nothing is sent to
a public host, the server binds to `127.0.0.1` only, and the whole thing degrades
honestly (clear error, not a 500) if a model is offline.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with the `nomic-embed-text` model
  (`ollama pull nomic-embed-text`)
- [LM Studio](https://lmstudio.ai) serving an OpenAI-compatible endpoint (or any
  compatible local LLM)

Install deps:

```bash
pip install -r requirements.txt
```

## Run the demo server

```bash
uvicorn coderag.main:app --host 127.0.0.1 --port 8090
```

Open <http://127.0.0.1:8090> for the ask UI, or call the API directly:

```bash
curl -s http://127.0.0.1:8090/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does the RAG pipeline retrieve relevant code?"}'
```

The index defaults to the project itself (so it can answer questions about its
own source). Override the root with `RAG_ROOT=/path/to/other/repo`.

### Configuration (environment)

| Variable             | Default                                   | Purpose                       |
|----------------------|-------------------------------------------|-------------------------------|
| `RAG_ROOT`           | this project's root                       | Codebase to index             |
| `RAG_EMBED_ENDPOINT` | `http://localhost:11434/api/embed`        | Ollama embed URL              |
| `RAG_EMBED_MODEL`    | `nomic-embed-text`                        | Embedding model               |
| `RAG_LLM_ENDPOINT`   | `http://localhost:1234/v1/chat/completions` | Chat completions URL        |
| `RAG_LLM_MODEL`      | `dirk-qwen3.8-27b@q4_k_s`                 | Chat model id                 |
| `RAG_TOP_K`          | `4`                                       | Passages retrieved per query  |
| `RAG_MAX_TOKENS`     | `512`                                     | Generation budget             |
| `RAG_HOST`           | `127.0.0.1`                               | Demo server bind address      |
| `RAG_PORT`           | `8090`                                    | Demo server port              |

### Run with Docker

```bash
docker compose up --build
```

The container binds the port to host loopback only, and reaches your local
Ollama / LM Studio through `host.docker.internal` (see `docker-compose.yml`).

> **Reasoning-model note:** LM Studio's local Qwen is a reasoning model. The
> synthesizer always sends `"reasoning_effort": "none"` so the answer lands in
> `content` (not `reasoning_content`) and returns quickly.

## Library usage

```python
from coderag import CodeRAG

rag = CodeRAG()                       # uses local defaults
result = rag.ask("How does retrieval work?")
print(result.answer)
for s in result.sources:
    print(s.path, s.symbol, s.start_line, s.end_line, s.score)
```

Custom endpoints / codebase:

```python
from pathlib import Path
from coderag import CodeRAG
from coderag.config import Settings

settings = Settings(
    root=Path("/path/to/repo"),
    embed_endpoint="http://localhost:11434/api/embed",
    embed_model="nomic-embed-text",
    llm_endpoint="http://localhost:1234/v1/chat/completions",
    llm_model="your-model-id",
    top_k=4,
    max_tokens=512,
    bind_host="127.0.0.1",
    bind_port=8090,
    static_dir=Path("frontend"),
)
rag = CodeRAG(settings)
```

The embedder and synthesizer are injectable (for tests or custom providers):

```python
rag = CodeRAG(settings, embedder=my_embedder, synthesizer=my_synthesizer)
```

## Tests

```bash
python -m pytest tests -q
```

The suite covers ingestion, retrieval math, the pipeline (with fake embedder /
synthesizer), and the HTTP API. No live model is required to run the tests.

## License

MIT. See [LICENSE](LICENSE).
