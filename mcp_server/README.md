# MCP Server — AI Analytics Assistant

The MCP server is a standalone service that exposes analytical tools to the LangGraph agents in the backend. It handles all data access, statistical computation, and Gemini API calls. The backend never touches the databases or the LLM directly — it goes through here.

It runs as its own Docker container and communicates with the backend over HTTP using the MCP Streamable HTTP transport.

---

## Directory Structure

```
mcp_server/
├── main.py                  Entry point. Registers all tools, handles startup/shutdown.
├── schemas.py               Pydantic models for every tool's input and output.
├── requirements.txt
├── Dockerfile
│
├── core/
│   ├── config.py            Settings loaded from environment variables.
│   ├── logging.py           Structured JSON logging via structlog.
│   ├── gemini_client.py     Gemini API wrapper with retry, rate limiting, and caching.
│   ├── rate_limiter.py      Redis sliding window counters for RPM, RPD, and TPM.
│   ├── cache.py             Redis get/set helpers with consistent key naming.
│   ├── guardrails.py        Input safety checks: injection, PII, SQL injection.
│   ├── langsmith_tracer.py  LangSmith trace context manager for every tool call.
│   └── tot_reasoner.py      Tree-of-Thought reasoning engine using Gemini.
│
├── db/
│   ├── redis_client.py      Async Redis connection pool.
│   ├── postgres.py          Async SQLAlchemy engine and session factory.
│   └── chromadb_client.py   ChromaDB async HTTP client.
│
└── tools/
    ├── vector_search.py     Semantic search over conversation history (ChromaDB).
    ├── metrics_query.py     Structured analytics queries against PostgreSQL.
    ├── anomaly_detector.py  Statistical anomaly detection (Z-score, IQR, Isolation Forest).
    ├── trend_analyzer.py    Trend analysis, seasonality detection, forecasting.
    ├── rag_tool.py          Full RAG pipeline: retrieve + generate with citations.
    ├── prompt_optimizer.py  Prompt performance tracking and variant generation.
    └── feedback_tool.py     User feedback recording and aggregated analytics.
```

---

## Tools

The LLM reads each tool's description at runtime and decides when to call it. Tool selection and parameter construction are handled by the LLM — tools execute deterministically.

| Tool | Function name | When the LLM uses it |
|---|---|---|
| Semantic search | `search_conversations` | Finding examples of a pattern or failure type |
| Metrics query | `get_metrics` | Quantitative questions: rates, volumes, counts |
| Anomaly detection | `find_anomalies` | "Why did X spike/drop?" investigations |
| Trend analysis | `get_trends` | Direction, momentum, forecasts over time |
| Top failure intents | `top_failure_intents` | "What is the bot struggling with?" |
| RAG query | `ask_with_rag` | Factual questions needing evidence from real data |
| Prompt optimizer | `improve_prompt` | When a prompt's performance score is low |
| Submit feedback | `submit_feedback` | After a user rates a response |
| Feedback analytics | `feedback_analytics` | Dashboard feedback charts and NPS |
| Deep analysis | `deep_analysis` | Multi-step reasoning for complex root cause questions |

---

## Data Flow

```
Agent calls a tool
    -> main.py receives the MCP request
    -> Guardrails check (prompt injection, PII, SQL injection)
    -> Tool function runs:
         - Checks Redis cache
         - Queries PostgreSQL or ChromaDB
         - Calls Gemini if needed (rate limiter -> retry loop -> cache result)
    -> Pydantic validates output
    -> LangSmith trace flushed in background
    -> JSON returned to agent
```

---

## Configuration

All settings are loaded from environment variables. Copy `.env.example` to `.env` from the project root and fill in the required values.

Required variables:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google AI Studio API key |
| `POSTGRES_PASSWORD` | PostgreSQL password |

Optional variables with defaults:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for generation |
| `GEMINI_RPM_LIMIT` | `10` | Requests per minute (free tier) |
| `GEMINI_RPD_LIMIT` | `1500` | Requests per day (free tier) |
| `GEMINI_TPM_LIMIT` | `250000` | Tokens per minute (free tier) |
| `GEMINI_MAX_RETRIES` | `5` | Max retry attempts on failure |
| `POSTGRES_HOST` | `postgres` | Hostname (use `localhost` outside Docker) |
| `REDIS_HOST` | `redis` | Hostname (use `localhost` outside Docker) |
| `CHROMADB_HOST` | `chromadb` | Hostname (use `localhost` outside Docker) |
| `MCP_PORT` | `8001` | Port the server listens on |
| `ENVIRONMENT` | `production` | `development` gives coloured logs |
| `DEBUG` | `false` | Enables SQL query logging |
| `LANGCHAIN_API_KEY` | _(empty)_ | Optional. If absent, tracing is silently skipped |
| `LANGCHAIN_PROJECT` | `ai-analytics-assistant` | LangSmith project name |

---

## Running

### With Docker (recommended)

Start infrastructure and the MCP server together from the project root:

```bash
cp .env.example .env
# Edit .env — fill in GEMINI_API_KEY and POSTGRES_PASSWORD

# Start infrastructure only
docker compose up postgres redis chromadb -d

# Wait for them to be healthy, then start the MCP server
docker compose up mcp_server
```

Or start everything at once:

```bash
docker compose up postgres redis chromadb mcp_server
```

The server will be available at `http://localhost:8001`.

### Without Docker (local development)

The infrastructure (Postgres, Redis, ChromaDB) still needs to run somewhere. The easiest option is to start just those containers and run the MCP server directly:

```bash
# Start only the data stores
docker compose up postgres redis chromadb -d

# Set up Python environment
cd mcp_server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Environment variables
export GEMINI_API_KEY=your_key
export POSTGRES_PASSWORD=changeme
export POSTGRES_HOST=localhost
export REDIS_HOST=localhost
export CHROMADB_HOST=localhost
export ENVIRONMENT=development

python main.py
```

### Confirm it is running

```bash
curl http://localhost:8001/health
```

You should get a JSON response like:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "postgres": true,
    "redis": true,
    "chromadb": true
  },
  "rate_limit_status": { ... },
  "uptime_seconds": 12.4
}
```

---

## Calling a Tool Manually

You can call any tool directly via HTTP to test it without the backend:

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "top_failure_intents",
      "arguments": {
        "time_range_start": "2025-01-01T00:00:00",
        "time_range_end": "2025-01-31T23:59:59",
        "top_n": 5,
        "include_examples": true
      }
    },
    "id": 1
  }'
```

---

## Fault Tolerance

**Rate limiting.** Before every Gemini call, `rate_limiter.py` checks three Redis counters: requests per minute, requests per day, and tokens per minute. If any counter is at its limit, the call waits (with async sleep) rather than failing. The wait time comes from the oldest entry in each Redis sorted set, so it is accurate.

**Retries.** Retryable errors (429, 503, "quota exceeded", "overloaded") trigger exponential backoff with random jitter. Non-retryable errors (bad API key, invalid request) fail immediately. Max retries is configurable via `GEMINI_MAX_RETRIES`.

**Caching.** Responses are cached in Redis keyed by a hash of the prompt and context. The same query within the TTL window returns from cache with no Gemini call. Embeddings are cached separately with a 24-hour TTL since they are deterministic.

**Guardrails.** Every tool input passes through `guardrails.py` before any processing. Prompt injection and SQL injection patterns block the request outright. PII is redacted silently and the sanitised text is used downstream.

---

## Development Notes

**Adding a new tool.** Create a file in `tools/`, define an async function that takes a Pydantic input and returns a Pydantic output, add the input/output schemas to `schemas.py`, then register the tool in `main.py` with a `@mcp.tool()` decorator and a clear description.

**Changing rate limits.** Set `GEMINI_RPM_LIMIT`, `GEMINI_RPD_LIMIT`, or `GEMINI_TPM_LIMIT` in `.env`. The rate limiter reads these from `settings` at runtime — no code change required.

**LangSmith.** Set `LANGCHAIN_API_KEY` in `.env`. All tool calls will appear in the LangSmith UI under the project name set in `LANGCHAIN_PROJECT`. If the key is absent, the server starts normally with no tracing.

**Logs.** In development (`ENVIRONMENT=development`), logs are coloured and printed to stdout. In production, they are JSON-formatted for log aggregators. Both include `service`, `version`, `env`, and structured fields per log event.

---

## Dependencies

The key libraries and what they do:

| Library | Purpose |
|---|---|
| `mcp[cli]` | MCP protocol implementation and Streamable HTTP server |
| `google-genai` | Gemini generation and embedding API |
| `sqlalchemy[asyncio]` + `asyncpg` | Async PostgreSQL queries |
| `redis[asyncio]` | Cache, rate limiting, connection pool |
| `chromadb` | Vector store for conversation embeddings |
| `pydantic` + `pydantic-settings` | Schema validation and settings management |
| `structlog` | Structured logging |
| `presidio-analyzer` + `presidio-anonymizer` | PII detection and redaction |
| `langsmith` | Observability and run tracing |
| `scikit-learn` | Isolation Forest for anomaly detection |
| `numpy` + `scipy` | Statistical computations in trend and anomaly tools |