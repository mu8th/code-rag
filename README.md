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
- **Embed**: deterministic feature-hashing vectors by default (offline, no
  model); optionally Ollama's `/api/embed` (nomic-embed-text) in live mode.
- **Retrieve**: a pure-numpy cosine index returns the top-k most relevant
  passages. This stage is real in both modes: simulation mode ranks by token
  overlap, which lands queries on the right symbols for code.
- **Synthesize**: rule-based grounded summary by default (names the top-ranked
  symbol and quotes each cited passage); optionally an OpenAI-compatible local
  chat endpoint in live mode. The primary endpoint defaults to LM Studio; if it
  is not running, the synthesizer falls back to a second local endpoint
  (Ollama's OpenAI-compatible API by default).

### Simulation mode (default)

The demo is meant to run 24/7 on a machine that does not keep a local LLM
loaded, so by default (`RAG_SIMULATE=1`) the pipeline runs **fully offline**:
no Ollama, no LM Studio, no network calls. Ingestion and retrieval are still
the real code paths; only the embedding and answer-generation steps are
deterministic stand-ins. Answers report `"model": "simulated"` so the UI can be
honest about what produced them. Set `RAG_SIMULATE=0` to use real local models.

## Why local?

Everything runs on the developer's own machine: nothing is sent to a public
host, the server binds to `127.0.0.1` only, and the whole thing degrades
honestly (clear error, not a 500) if a live-mode model endpoint is offline.

## Requirements

- Python 3.11+ (that is all simulation mode needs)
- For live mode (`RAG_SIMULATE=0`):
  - [Ollama](https://ollama.com) with `nomic-embed-text`
    (`ollama pull nomic-embed-text`) and a chat model, e.g. `qwen3:4b`
    (`ollama pull qwen3:4b`)
  - Optionally [LM Studio](https://lmstudio.ai) (or any OpenAI-compatible
    local LLM) as the primary answer endpoint; if it is down, Ollama's chat
    model is used instead

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
| `RAG_SIMULATE`       | `1`                                       | Offline simulation mode (no models); `0` for live local models |
| `RAG_ROOT`           | this project's root                       | Codebase to index             |
| `RAG_EMBED_ENDPOINT` | `http://localhost:11434/api/embed`        | Ollama embed URL              |
| `RAG_EMBED_MODEL`    | `nomic-embed-text`                        | Embedding model               |
| `RAG_LLM_ENDPOINT`   | `http://localhost:1234/v1/chat/completions` | Primary chat URL (LM Studio)  |
| `RAG_LLM_MODEL`      | `dirk-qwen3.8-27b@q4_k_s`                 | Primary chat model id       |
| `RAG_LLM_FALLBACK_ENDPOINT` | `http://localhost:11434/v1/chat/completions` | Fallback chat URL (Ollama); tried only if the primary refuses connections |
| `RAG_LLM_FALLBACK_MODEL`  | `qwen3:4b`                        | Fallback model id           |
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
    simulate=False,  # use the real local model endpoints above
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
