"""
MCP Tool: optimize_prompt

Tracks prompt performance over time and uses Gemini to generate
improved variants based on historical data and user feedback.

Flow:
  1. Load historical performance for this prompt key
  2. Optionally load user feedback that references this prompt
  3. Use Gemini to analyse what's working and what isn't
  4. Generate N improved variants
  5. Store variants in PostgreSQL for A/B testing
  6. Return variants with rationale

LangSmith integration: records prompt versions so you can compare
performance across variants in the LangSmith UI.
"""

import time
import uuid
from datetime import datetime, timedelta

from sqlalchemy import text

from db.postgres import get_db
from core.gemini_client import generate
from core.guardrails import check_input, GuardrailError
from core.langsmith_tracer import trace_tool
from core.config import settings
from core.logging import get_logger
from schemas import (
    PromptOptimizeInput,
    PromptOptimizeOutput,
    PromptVariant,
    PromptPerformanceRecord,
)

logger = get_logger(__name__)


_OPTIMIZATION_PROMPT = """\
You are an expert prompt engineer for LLM-based analytics systems.

Current prompt template:
---
{current_template}
---

Historical performance data:
- Average score: {avg_score:.2f}/1.0
- Total uses: {usage_count}
- Optimization goal: {goal}

Recent user feedback:
{feedback_summary}

Issues identified:
{issues}

Generate exactly {n_variants} improved versions of this prompt.
Each variant should address the identified issues while optimizing for {goal}.

Format your response as:

VARIANT_1_DESCRIPTION: <brief description of the improvement>
VARIANT_1_TEMPLATE:
<full prompt template>
END_VARIANT_1

VARIANT_2_DESCRIPTION: <brief description>
VARIANT_2_TEMPLATE:
<full prompt template>
END_VARIANT_2
...

RECOMMENDATION: VARIANT_<N> because <reason>
RATIONALE: <overall explanation of the optimization strategy>
"""

_ISSUE_ANALYSIS_PROMPT = """\
Analyze this prompt template for potential issues based on its performance data.

Prompt:
{template}

Performance: avg score {avg_score:.2f}/1.0 with {usage_count} uses.
Goal: optimize for {goal}

List 2-4 specific, actionable issues with this prompt. Be concrete.
Each issue on its own line starting with "- ".
"""


async def optimize_prompt(params: PromptOptimizeInput) -> PromptOptimizeOutput:
    """
    Analyse prompt performance and generate improved variants.
    """
    async with trace_tool(
        "optimize_prompt",
        inputs={"prompt_key": params.prompt_key, "goal": params.optimization_goal},
    ) as run:
        t0 = time.perf_counter()

        # ── Guardrail on prompt template ──────────────────────────────────────
        guard = await check_input(params.current_template, context="prompt_template")
        if not guard.is_safe:
            raise GuardrailError(guard)

        # ── Load historical performance ───────────────────────────────────────
        perf = await _load_performance(params.prompt_key)
        avg_score = perf.get("avg_score", 0.5)
        usage_count = perf.get("usage_count", 0)

        # ── Load feedback if requested ────────────────────────────────────────
        feedback_summary = "No feedback data available."
        if params.use_feedback_data:
            feedback_summary = await _load_feedback_summary(params.prompt_key)

        # ── Analyse current issues ────────────────────────────────────────────
        issues = await generate(
            prompt=_ISSUE_ANALYSIS_PROMPT.format(
                template=params.current_template,
                avg_score=avg_score,
                usage_count=usage_count,
                goal=params.optimization_goal,
            ),
            temperature=0.3,
            use_cache=True,
        )

        # ── Generate variants ─────────────────────────────────────────────────
        response = await generate(
            prompt=_OPTIMIZATION_PROMPT.format(
                current_template=params.current_template,
                avg_score=avg_score,
                usage_count=usage_count,
                goal=params.optimization_goal,
                feedback_summary=feedback_summary,
                issues=issues,
                n_variants=params.n_variants,
            ),
            temperature=0.6,
            use_cache=False,    # Optimizations should always be fresh
        )

        variants, recommended_id, rationale = _parse_variants_response(
            response, params.n_variants
        )

        # ── Store variants in PostgreSQL ──────────────────────────────────────
        await _store_variants(params.prompt_key, variants)

        output = PromptOptimizeOutput(
            prompt_key=params.prompt_key,
            original_template=params.current_template,
            variants=variants,
            optimization_rationale=rationale,
            recommended_variant_id=recommended_id or (variants[0].variant_id if variants else ""),
            expected_improvement=(
                f"Targeting improvement in {params.optimization_goal} "
                f"from current baseline of {avg_score:.2f}/1.0"
            ),
        )

        run.add_output(output)
        logger.info(
            "prompt_optimization_complete",
            prompt_key=params.prompt_key,
            variants=len(variants),
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return output


async def record_prompt_performance(record: PromptPerformanceRecord) -> None:
    """
    Record a single prompt execution's performance score.
    Called by the backend after each agent run completes.
    """
    async with get_db() as db:
        await db.execute(text("""
            INSERT INTO prompt_performances
                (prompt_key, variant_id, score, feedback_text, latency_ms, tokens_used, recorded_at)
            VALUES
                (:prompt_key, :variant_id, :score, :feedback_text, :latency_ms, :tokens_used, NOW())
            ON CONFLICT DO NOTHING
        """), {
            "prompt_key": record.prompt_key,
            "variant_id": record.variant_id,
            "score": record.score,
            "feedback_text": record.feedback_text,
            "latency_ms": record.latency_ms,
            "tokens_used": record.tokens_used,
        })
    logger.debug(
        "prompt_performance_recorded",
        prompt_key=record.prompt_key,
        score=record.score,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_performance(prompt_key: str) -> dict:
    async with get_db() as db:
        row = await db.execute(text("""
            SELECT
                AVG(score)   AS avg_score,
                COUNT(*)     AS usage_count
            FROM prompt_performances
            WHERE prompt_key = :key
              AND recorded_at > NOW() - INTERVAL '30 days'
        """), {"key": prompt_key})
        result = row.fetchone()
        if result and result.usage_count:
            return {
                "avg_score": float(result.avg_score or 0.5),
                "usage_count": int(result.usage_count),
            }
    return {"avg_score": 0.5, "usage_count": 0}


async def _load_feedback_summary(prompt_key: str) -> str:
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT feedback_text, score
            FROM prompt_performances
            WHERE prompt_key = :key
              AND feedback_text IS NOT NULL
              AND recorded_at > NOW() - INTERVAL '14 days'
            ORDER BY recorded_at DESC
            LIMIT 10
        """), {"key": prompt_key})
        feedbacks = rows.fetchall()

    if not feedbacks:
        return "No recent user feedback available."

    lines = [
        f"- (score={row.score:.2f}) {row.feedback_text}"
        for row in feedbacks
    ]
    return "\n".join(lines)


def _parse_variants_response(
    response: str, n_variants: int
) -> tuple[list[PromptVariant], str, str]:
    """Parse structured variant response from Gemini."""
    variants: list[PromptVariant] = []
    recommended_id = ""
    rationale = ""

    import re

    # Extract rationale
    rat_match = re.search(r"RATIONALE:\s*(.+?)(?=VARIANT_|\Z)", response, re.DOTALL)
    if rat_match:
        rationale = rat_match.group(1).strip()

    # Extract recommendation
    rec_match = re.search(r"RECOMMENDATION:\s*VARIANT_(\d+)", response, re.I)

    for i in range(1, n_variants + 1):
        desc_match = re.search(
            rf"VARIANT_{i}_DESCRIPTION:\s*(.+?)(?=VARIANT_{i}_TEMPLATE:)", response, re.DOTALL
        )
        template_match = re.search(
            rf"VARIANT_{i}_TEMPLATE:\s*(.+?)(?=END_VARIANT_{i})", response, re.DOTALL
        )

        if not template_match:
            continue

        variant_id = str(uuid.uuid4())[:12]
        description = desc_match.group(1).strip() if desc_match else f"Variant {i}"
        template = template_match.group(1).strip()

        variants.append(PromptVariant(
            variant_id=variant_id,
            template=template,
            description=description,
            created_at=datetime.utcnow(),
        ))

        # Map recommendation to variant_id
        if rec_match and int(rec_match.group(1)) == i:
            recommended_id = variant_id

    # Fallback: if parsing failed, return the whole response as one variant
    if not variants:
        variant_id = str(uuid.uuid4())[:12]
        variants.append(PromptVariant(
            variant_id=variant_id,
            template=response,
            description="Generated variant (parsing fallback)",
            created_at=datetime.utcnow(),
        ))
        recommended_id = variant_id
        rationale = rationale or "Unable to parse structured response."

    return variants, recommended_id, rationale


async def _store_variants(prompt_key: str, variants: list[PromptVariant]) -> None:
    async with get_db() as db:
        for v in variants:
            await db.execute(text("""
                INSERT INTO prompt_variants
                    (variant_id, prompt_key, template, description, created_at)
                VALUES
                    (:vid, :pkey, :template, :desc, :created_at)
                ON CONFLICT (variant_id) DO NOTHING
            """), {
                "vid": v.variant_id,
                "pkey": prompt_key,
                "template": v.template,
                "desc": v.description,
                "created_at": v.created_at or datetime.utcnow(),
            })