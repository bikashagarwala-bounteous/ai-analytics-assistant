"""
MCP Tool: rag_query

Full Retrieval-Augmented Generation pipeline:
  1. Embed the user's query (Gemini embedding-001)
  2. Retrieve top-k semantically similar conversation chunks (ChromaDB)
  3. Build a grounded context prompt from retrieved chunks
  4. Generate an answer using Gemini, citing source IDs
  5. Return structured output with answer + sources

This is the tool the LLM should use when:
- The user asks a specific factual question about conversation history
- The analyst agent needs evidence to support a hypothesis
- "Why did X happen?" questions that need real examples
"""

import time
from datetime import datetime

from db.chromadb_client import query_collection
from core.gemini_client import embed_query, generate
from core.guardrails import check_input, check_output, GuardrailError
from core.langsmith_tracer import trace_tool
from core.cache import rag_cache_key, cache_get, cache_set
from core.config import settings
from core.logging import get_logger
from schemas import RAGQueryInput, RAGQueryOutput, RAGSource

logger = get_logger(__name__)


_RAG_SYSTEM_PROMPT = """\
You are an expert business analytics assistant. You answer questions about chatbot
conversation data based ONLY on the provided context chunks.

Rules:
- Ground every claim in the provided context. Do not hallucinate.
- If the context does not contain enough information, say so explicitly.
- Reference specific message IDs when citing evidence (e.g. [msg_abc123]).
- Be concise and business-focused. Avoid technical jargon unless necessary.
- If you see contradictory evidence in the context, note the contradiction.
"""

_RAG_USER_PROMPT = """\
Question: {query}

Context (retrieved conversation chunks, most similar first):
{context}

Answer the question based on the context above. Cite message IDs as [msg_id] inline.
"""


async def rag_query(params: RAGQueryInput) -> RAGQueryOutput:
    """
    Full RAG pipeline: retrieve relevant context → generate grounded answer.

    Flow:
    1. Guardrail check on the query
    2. Cache check
    3. Embed query → ChromaDB search
    4. Build context prompt from retrieved chunks
    5. Gemini generate with grounded context
    6. Guardrail check on output (PII redaction)
    7. Cache result
    """
    async with trace_tool(
        "rag_query",
        inputs={"query": params.query[:200], "collection": params.collection},
    ) as run:
        t0 = time.perf_counter()

        # ── Guardrail on input ────────────────────────────────────────────────
        guard_result = await check_input(params.query, context="rag_query")
        if not guard_result.is_safe:
            raise GuardrailError(guard_result)
        safe_query = guard_result.sanitized_text

        # ── Cache check ───────────────────────────────────────────────────────
        cache_key = rag_cache_key(
            f"rag:{safe_query}:{params.collection}:{params.top_k}"
        )
        cached = await cache_get(cache_key)
        if cached:
            logger.debug("rag_cache_hit", query=safe_query[:50])
            return RAGQueryOutput(**cached)

        # ── Retrieval ─────────────────────────────────────────────────────────
        t_retrieve = time.perf_counter()
        query_vector = await embed_query(safe_query)

        fetch_n = params.top_k * 4 if params.time_range else params.top_k
        raw_results = await query_collection(
            collection_name=params.collection,
            query_embeddings=[query_vector],
            n_results=fetch_n,
            where=None,
        )
        retrieval_ms = (time.perf_counter() - t_retrieve) * 1000

        # ── Parse retrieved sources ───────────────────────────────────────────
        sources: list[RAGSource] = []
        context_chunks: list[str] = []

        if raw_results and raw_results.get("ids"):
            ids = raw_results["ids"][0]
            documents = raw_results["documents"][0]
            metadatas = raw_results["metadatas"][0]
            distances = raw_results["distances"][0]

            for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                similarity = 1.0 - (dist / 2.0)
                if similarity < 0.4:
                    continue

                raw_ts = meta.get("timestamp") or meta.get("timestamp_unix")
                try:
                    ts = (datetime.fromtimestamp(float(raw_ts))
                          if isinstance(raw_ts, (int, float))
                          else datetime.fromisoformat(str(raw_ts)))
                except Exception:
                    ts = datetime.utcnow()

                # Python-side time range filter
                if params.time_range and not (params.time_range.start <= ts <= params.time_range.end):
                    continue

                if len(sources) >= params.top_k:
                    break

                source = RAGSource(
                    message_id=doc_id,
                    content=doc,
                    similarity=round(similarity, 4),
                    timestamp=ts,
                    intent=meta.get("intent"),
                )
                sources.append(source)
                context_chunks.append(
                    f"[{doc_id}] (similarity={similarity:.2f}): {doc}"
                )

        if not sources:
            logger.warning("rag_no_relevant_chunks", query=safe_query[:80])
            return RAGQueryOutput(
                query=safe_query,
                answer=(
                    "I couldn't find relevant conversation data to answer this question. "
                    "The question may be outside the scope of the available data, "
                    "or the time range may have no matching conversations."
                ),
                sources=[],
                confidence=0.0,
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=0.0,
            )

        # ── Generation ────────────────────────────────────────────────────────
        t_gen = time.perf_counter()
        context_text = "\n\n".join(context_chunks[:params.top_k])

        answer = await generate(
            prompt=_RAG_USER_PROMPT.format(
                query=safe_query,
                context=context_text,
            ),
            system_prompt=_RAG_SYSTEM_PROMPT,
            temperature=params.temperature,
            use_cache=False,    # RAG answers should always reflect current retrieval
            estimated_input_tokens=len(context_text) // 4 + 200,
        )
        generation_ms = (time.perf_counter() - t_gen) * 1000

        # ── Output guardrail (PII redaction) ──────────────────────────────────
        answer = await check_output(answer)

        # ── Confidence = avg similarity of top sources ────────────────────────
        confidence = (
            sum(s.similarity for s in sources) / len(sources)
            if sources else 0.0
        )

        output = RAGQueryOutput(
            query=safe_query,
            answer=answer,
            sources=sources if params.include_sources else [],
            confidence=round(confidence, 4),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
        )

        # ── Cache result ──────────────────────────────────────────────────────
        await cache_set(
            cache_key,
            output.model_dump(mode="json"),
            settings.cache_ttl_rag_retrieval,
        )

        run.add_output(output)
        run.add_metadata(
            sources_retrieved=len(sources),
            confidence=confidence,
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
        )

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "rag_query_complete",
            query=safe_query[:60],
            sources=len(sources),
            confidence=round(confidence, 3),
            total_ms=round(total_ms, 1),
        )
        return output