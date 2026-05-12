# AI Business Analytics Assistant

A multi-agent system that analyses chatbot conversation data and answers natural-language questions about performance. Ask things like "Why did engagement drop last week?" or "What are the top failure intents?" and the system retrieves data, detects anomalies, reasons over it, and streams back a grounded answer.

---

## Architecture

```
Streamlit UI  (port 8501)
     |
     | HTTP
     v
FastAPI Backend  (port 8080)
     |                        |
     | LangGraph pipeline     | /dashboard, /report, /feedback
     v                        v
  Router → Analyst → Summary    PostgreSQL  (port 5432)
                |
                | MCP tool calls (HTTP)
                v
          MCP Server  (port 8001)
               |           |           |
               v           v           v
          PostgreSQL    Redis       ChromaDB
          (metrics)   (cache,     (embeddings)
                      rate limit)

Background Indexer (no port)
  Reads Redis queue → embeds via Gemini → writes to ChromaDB
```

**Six containers run via Docker Compose:**

| Container | Role |
|---|---|
| `backend` | FastAPI app. Hosts the LangGraph pipeline and `/chat`, `/analyze`, `/techniques` routes. |
| `mcp_server` | Standalone MCP service. Exposes tools the agents call over HTTP. |
| `indexer` | Background worker. Drains the Redis message queue and keeps ChromaDB up to date. |
| `frontend` | Streamlit app with Dashboard, Chat, and Prompt Lab pages. |
| `postgres` | Conversation logs, sessions, agent runs, reports, feedback, prompt variants. |
| `redis` | Response cache, embedding cache, rate-limit counters, indexing queue. |
| `chromadb` | Vector embeddings for RAG and semantic search. |

---

## Pipeline

Queries flow through a three-stage LangGraph pipeline:

1. **Router** — classifies the query (greeting / identity / capability / analytics / off_topic). Non-analytics queries return a canned response immediately; analytics queries proceed to the analyst. Retries up to three times on transient Gemini errors before defaulting to analytics.

2. **Analyst Agent** — calls Gemini to produce a JSON tool execution plan, runs the MCP tools (in parallel where the plan allows), then calls Gemini again to synthesize structured `AnalystFindings`. Supports Tree-of-Thought synthesis for complex root-cause questions.

3. **Summary Agent** — loads the active prompt variant from PostgreSQL, calls Gemini to stream a business-readable response, and persists the completed report. Falls back to blocking generation if streaming fails.

---

## Features

**MCP Server tools**
- `get_metrics` — engagement rate, failure rate, resolution rate, message volume, session duration, unique users, avg turns, intent distribution
- `find_anomalies` — Z-score, IQR, or Isolation Forest detection with severity levels
- `get_trends` — linear regression, change-point detection, seasonality analysis, optional forecasting
- `top_failure_intents` — ranked failing intents with sample messages and trend direction
- `ask_with_rag` — embed → retrieve → generate with source citations
- `search_conversations` — semantic similarity search over conversation history
- `improve_prompt` — analyse prompt performance and generate improved variants
- `submit_feedback` / `feedback_analytics` — user ratings synced to the dashboard
- `deep_analysis` — Tree-of-Thought reasoning exposed as a callable tool

**Techniques Playground** (`/techniques/run`)

Experiment with five prompting strategies directly from the UI or API:

| Technique | How it works |
|---|---|
| `zero_shot` | System prompt + query only |
| `few_shot` | Inline input/output examples before the query |
| `chain_of_thought` | Forces step-by-step reasoning before the final answer |
| `react` | Thought → Action → Observation loop (simulated, no live tools) |
| `tree_of_thoughts` | N branches in parallel; a judge picks the best answer |

Each run returns token counts and USD cost.

**Fault tolerance**
- Router, analyst planning, analyst synthesis, and summary all retry on Gemini 500/503 with exponential backoff (up to 4 attempts)
- Redis response cache (1 hour) skips the full pipeline for repeated queries
- Embedding cache (24 hours) — each unique text embedded once
- MCP tool call retry with dead-letter queue in the indexer
- SSE heartbeat every 20 s keeps long analyst runs from timing out in the browser

**Safety**
- Input guardrails on every MCP tool: prompt injection, SQL injection, PII detection (Presidio)
- Output PII redaction on all generated text
- All SQL parameterised

**Observability**
- Structured JSON logs (dev: coloured); per-stage retry events logged at `WARNING`
- Agent run history in PostgreSQL with tool call records and per-stage timing
- LangSmith tracing (optional — omit `LANGCHAIN_API_KEY` to disable)
- `/health` endpoints on backend and MCP server

---

## Project Structure

```
ai-analytics-assistant/
├── .env.example
├── docker-compose.yml
│
├── mcp_server/
│   ├── main.py              Tool registration and server entry point
│   ├── schemas.py           Pydantic models for all tool inputs/outputs
│   ├── core/
│   │   ├── config.py
│   │   ├── gemini_client.py Gemini wrapper with retry, rate limiting, caching
│   │   ├── rate_limiter.py  Redis sliding window rate limiter
│   │   ├── cache.py
│   │   ├── guardrails.py    Input safety checks
│   │   ├── langsmith_tracer.py
│   │   └── tot_reasoner.py  Tree-of-Thought engine
│   ├── db/
│   │   ├── postgres.py
│   │   ├── redis_client.py
│   │   └── chromadb_client.py
│   └── tools/
│       ├── vector_search.py
│       ├── metrics_query.py
│       ├── anomaly_detector.py
│       ├── trend_analyzer.py
│       ├── rag_tool.py
│       ├── prompt_optimizer.py
│       └── feedback_tool.py
│
├── backend/
│   ├── main.py
│   ├── agents/
│   │   ├── state.py         LangGraph shared state and data models
│   │   ├── graph.py         Pipeline assembly, router node, run functions
│   │   ├── analyst_agent.py Plans and executes MCP tool calls, synthesizes findings
│   │   └── summary_agent.py Streams grounded response, stores report
│   ├── api/routes/
│   │   ├── chat.py          POST /chat — SSE streaming
│   │   ├── analyze.py       POST /analyze — structured findings
│   │   ├── report.py        GET /report, GET /dashboard, POST /feedback
│   │   └── techniques.py    POST /techniques/run — prompting technique playground
│   ├── db/
│   │   ├── connections.py
│   │   ├── models.py        SQLAlchemy ORM models
│   │   └── migrations/
│   │       └── init.sql     Full schema, run once on first container start
│   ├── schemas/
│   │   └── api.py           Request/response Pydantic models
│   ├── services/
│   │   ├── mcp_client.py    HTTP client for MCP tool calls
│   │   ├── session_service.py
│   │   └── dashboard_service.py
│   └── core/
│       ├── config.py
│       └── logging.py
│
├── indexer/
│   ├── main.py              Queue drain loop, embedding, ChromaDB upsert
│   └── config.py
│
└── frontend/
    ├── app.py               Entry point and navigation
    ├── pages/
    │   ├── dashboard.py     KPIs, trend charts, intent breakdown, feedback
    │   ├── chat.py          Streaming chat with feedback widget
    │   └── prompt_lab.py    Single test, A/B compare, optimizer, Techniques Playground
    └── components/
        ├── api_client.py    Backend HTTP client with SSE streaming
        └── charts.py        Plotly chart builders
```

---

## Setup

### Requirements

- Docker and Docker Compose
- A Google AI Studio API key (free tier works)

### First run

```bash
git clone <repo-url>
cd ai-analytics-assistant

cp .env.example .env
```

Open `.env` and set:

```
GEMINI_API_KEY=your_key_here
POSTGRES_PASSWORD=choose_any_password
```

Everything else has working defaults for local Docker.

```bash
docker compose up
```

On first start Docker builds all images (a few minutes). PostgreSQL runs `init.sql` automatically to create all tables.

Open the UI at **http://localhost:8501**.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Google AI Studio key |
| `POSTGRES_PASSWORD` | _(required)_ | PostgreSQL password |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used by all agents |
| `GEMINI_RPM_LIMIT` | `10` | Requests per minute cap |
| `GEMINI_RPD_LIMIT` | `1500` | Requests per day cap |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith tracing — omit to disable |
| `LANGCHAIN_PROJECT` | `ai-analytics-assistant` | LangSmith project name |
| `ENVIRONMENT` | `production` | Set to `development` for coloured logs |
| `DEBUG` | `false` | Set to `true` for SQL query logs |

---

## Running without Docker

```bash
# Start data stores only
docker compose up postgres redis chromadb -d

# MCP server
cd mcp_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key POSTGRES_PASSWORD=your_password \
       POSTGRES_HOST=localhost REDIS_HOST=localhost CHROMADB_HOST=localhost
python main.py

# Backend (new terminal)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key POSTGRES_PASSWORD=your_password \
       POSTGRES_HOST=localhost REDIS_HOST=localhost \
       MCP_SERVER_URL=http://localhost:8001/mcp
python main.py

# Indexer (new terminal)
cd indexer && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key POSTGRES_PASSWORD=your_password \
       POSTGRES_HOST=localhost REDIS_HOST=localhost CHROMADB_HOST=localhost
python main.py

# Frontend (new terminal)
cd frontend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8080
streamlit run app.py
```

---

## Service ports

| Service | Port |
|---|---|
| Streamlit UI | 8501 |
| FastAPI backend | 8080 |
| MCP server | 8001 |
| ChromaDB | 8000 |
| PostgreSQL | 5432 |
| Redis | 6379 |

---

## Health checks

```bash
curl http://localhost:8080/health   # Backend
curl http://localhost:8001/health   # MCP server
```

---

## Calling the API directly

**Non-streaming analysis:**
```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the top failure intents this week?"}'
```

**Streaming chat:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did engagement drop?", "stream": true}'
```

**Techniques playground:**
```bash
curl -X POST http://localhost:8080/techniques/run \
  -H "Content-Type: application/json" \
  -d '{"technique": "chain_of_thought", "user_query": "Explain engagement rate", "model": "gemini-2.5-flash"}'
```

**Dashboard metrics:**
```bash
curl http://localhost:8080/dashboard?days=7
```

---

## How data flows end to end

1. Frontend calls `POST /chat` with `stream: true`
2. Backend creates or looks up the session in PostgreSQL
3. User message stored in `chat_messages` and pushed to Redis indexing queue
4. LangGraph pipeline: router classifies → analyst plans and runs tools → summary streams response
5. Analyst calls Gemini to plan tool calls, executes them against the MCP server
6. MCP tools query PostgreSQL and ChromaDB; call Gemini for embeddings or generation
7. Results cached in Redis; analyst synthesizes structured findings
8. Summary agent loads active prompt from PostgreSQL, streams response via SSE
9. Completed response stored in `chat_messages` and `reports`; prompt performance recorded

**Background (indexer):**
- Drains the Redis queue, embeds messages via Gemini, upserts into ChromaDB
- On startup scans the last 24 hours for any messages not yet indexed
- Failed attempts retried up to 3 times before moving to a dead-letter list

---

## Gemini free tier

The system targets Gemini free tier limits (10 RPM, 1500 RPD, 250K TPM):

- Sliding window counters in Redis gate every Gemini call
- Responses cached in Redis (1 hour) — repeated queries skip the API entirely
- Embeddings cached for 24 hours — each unique text embedded once
- MCP server and indexer share the same Redis counters

Increase limits in `.env` when using a paid API key.

---

## Adding a new tool

1. Add Pydantic models to `mcp_server/schemas.py`
2. Create the tool in `mcp_server/tools/your_tool.py`
3. Export it from `mcp_server/tools/__init__.py`
4. Register it in `mcp_server/main.py` with `@mcp.tool()` and a clear description — the analyst's planning prompt will discover it automatically

---

## Tech stack

| Component | Library |
|---|---|
| LLM and embeddings | `google-genai` (Gemini 2.5 Flash) |
| Agent orchestration | `langgraph` |
| MCP server | `mcp[cli]` with Streamable HTTP transport |
| API framework | `fastapi` + `uvicorn` |
| UI | `streamlit` |
| Async ORM | `sqlalchemy[asyncio]` + `asyncpg` |
| Vector store | `chromadb` |
| Cache and queue | `redis[asyncio]` |
| Validation | `pydantic` + `pydantic-settings` |
| PII detection | `presidio-analyzer` + `presidio-anonymizer` |
| Observability | `langsmith` + `structlog` |
| Charts | `plotly` |
