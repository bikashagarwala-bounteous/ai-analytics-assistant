"""
Tree-of-Thought reasoning engine (Yao et al., 2023).
Generates thought branches, scores each with Gemini, prunes low-scoring paths, returns the best.
Strategies: best_first (default), bfs, dfs.
"""

import asyncio
import time
import uuid
from typing import Any

from core.gemini_client import generate
from core.logging import get_logger
from schemas import ToTReasoningInput, ToTReasoningOutput, ToTThought

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_THOUGHT_GEN_PROMPT = """\
You are reasoning step-by-step about an analytics problem.

Problem: {problem}
Context Data: {context}
Current reasoning so far: {path_so_far}

Generate exactly {k} distinct next reasoning steps.
Each step should explore a DIFFERENT angle or hypothesis.

Format your response as:
THOUGHT_1: <reasoning step>
THOUGHT_2: <reasoning step>
...
THOUGHT_{k}: <reasoning step>

Be specific, analytical, and grounded in the provided context data.
"""

_THOUGHT_EVAL_PROMPT = """\
You are evaluating the quality of a reasoning step for an analytics problem.

Problem: {problem}
Reasoning path so far: {path_so_far}
Candidate next thought: {thought}

Rate this reasoning step on a scale of 0.0 to 1.0 based on:
- Relevance to the problem (0.4 weight)
- Logical soundness (0.3 weight)  
- Progress toward a conclusion (0.3 weight)

Respond with ONLY a number between 0.0 and 1.0 (e.g. 0.75).
No other text.
"""

_TERMINAL_CHECK_PROMPT = """\
Given this reasoning path for the problem below, determine if we have reached a conclusion.

Problem: {problem}
Reasoning path: {path}

Is this a complete, actionable conclusion? Reply with YES or NO only.
"""

_FINAL_ANSWER_PROMPT = """\
You have completed a multi-step reasoning analysis. Synthesize the following reasoning path
into a clear, concise final answer for a business analyst.

Problem: {problem}
Context: {context}
Reasoning path:
{path}

Write a final analytical conclusion that:
1. Directly answers the problem
2. Cites the key reasoning steps that led to the conclusion
3. Notes confidence level and any caveats
4. Is written for a non-technical business audience

Final Answer:
"""


# ── Engine ────────────────────────────────────────────────────────────────────

async def reason(params: ToTReasoningInput) -> ToTReasoningOutput:
    """
    Execute Tree-of-Thought reasoning over the given problem.

    The LLM is called for:
    - Thought generation (branching_factor calls per level)
    - Thought evaluation (1 call per thought)
    - Terminal state detection (1 call per leaf)
    - Final answer synthesis (1 call)

    All calls go through the Gemini client with full rate limiting and retry logic.
    """
    t0 = time.perf_counter()

    all_thoughts: dict[str, ToTThought] = {}
    root_id = "root"
    explored_count = 0

    # ── Dispatch to traversal strategy ───────────────────────────────────────
    if params.strategy == "best_first":
        best_path = await _best_first_search(
            params, all_thoughts, root_id
        )
    elif params.strategy == "bfs":
        best_path = await _bfs_search(params, all_thoughts, root_id)
    else:
        best_path = await _dfs_search(params, all_thoughts, root_id)

    explored_count = len(all_thoughts)

    # ── Synthesize final answer from best path ────────────────────────────────
    path_text = "\n".join(
        f"Step {i+1}: {t.content}"
        for i, t in enumerate(best_path)
    )

    final_answer = await generate(
        prompt=_FINAL_ANSWER_PROMPT.format(
            problem=params.problem,
            context=params.context[:2000] if params.context else "No additional context",
            path=path_text,
        ),
        temperature=0.3,
        use_cache=False,
    )

    confidence = (
        sum(t.score for t in best_path) / len(best_path)
        if best_path else 0.0
    )

    duration_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "tot_reasoning_complete",
        strategy=params.strategy,
        thoughts_explored=explored_count,
        path_depth=len(best_path),
        confidence=round(confidence, 3),
        duration_ms=round(duration_ms, 1),
    )

    return ToTReasoningOutput(
        problem=params.problem,
        best_answer=final_answer,
        reasoning_path=best_path,
        all_thoughts=list(all_thoughts.values()),
        confidence=round(confidence, 4),
        total_thoughts_explored=explored_count,
        reasoning_duration_ms=round(duration_ms, 2),
    )


# ── Traversal strategies ──────────────────────────────────────────────────────

async def _best_first_search(
    params: ToTReasoningInput,
    all_thoughts: dict[str, ToTThought],
    root_id: str,
) -> list[ToTThought]:
    """
    Priority queue: always expand the leaf with the highest score.
    Best for problems where one line of reasoning is clearly superior.
    """
    import heapq

    # Priority queue: (-score, thought_id, path)
    # Negative score because heapq is a min-heap
    frontier: list[tuple[float, str, list[ToTThought]]] = []
    heapq.heappush(frontier, (0.0, root_id, []))

    best_terminal_path: list[ToTThought] = []
    best_terminal_score = -1.0

    while frontier and len(all_thoughts) < params.branching_factor ** params.max_depth:
        neg_score, current_id, current_path = heapq.heappop(frontier)

        if len(current_path) >= params.max_depth:
            score = -neg_score
            if score > best_terminal_score:
                best_terminal_score = score
                best_terminal_path = current_path
            continue

        # Generate children
        path_text = _path_to_text(current_path)
        children = await _generate_thoughts(
            params.problem,
            params.context,
            path_text,
            params.branching_factor,
        )

        child_scores = await _evaluate_thoughts_parallel(
            params.problem, path_text, children
        )

        for thought_text, score in zip(children, child_scores):
            thought = ToTThought(
                thought_id=str(uuid.uuid4())[:8],
                depth=len(current_path) + 1,
                content=thought_text,
                score=score,
                parent_id=current_path[-1].thought_id if current_path else None,
            )
            all_thoughts[thought.thought_id] = thought

            new_path = current_path + [thought]

            # Check if this is a terminal thought
            is_terminal = await _is_terminal(params.problem, new_path)
            thought.is_terminal = is_terminal

            if is_terminal:
                if score > best_terminal_score:
                    best_terminal_score = score
                    best_terminal_path = new_path
            else:
                heapq.heappush(frontier, (-score, thought.thought_id, new_path))

    return best_terminal_path or _get_best_partial_path(all_thoughts)


async def _bfs_search(
    params: ToTReasoningInput,
    all_thoughts: dict[str, ToTThought],
    root_id: str,
) -> list[ToTThought]:
    """Level-by-level exploration. Good for shallow reasoning trees."""
    current_level: list[list[ToTThought]] = [[]]   # Start with empty path
    best_path: list[ToTThought] = []
    best_score = -1.0

    for depth in range(params.max_depth):
        next_level: list[list[ToTThought]] = []

        for path in current_level:
            path_text = _path_to_text(path)
            children = await _generate_thoughts(
                params.problem, params.context, path_text, params.branching_factor
            )
            scores = await _evaluate_thoughts_parallel(params.problem, path_text, children)

            for thought_text, score in zip(children, scores):
                thought = ToTThought(
                    thought_id=str(uuid.uuid4())[:8],
                    depth=depth + 1,
                    content=thought_text,
                    score=score,
                    parent_id=path[-1].thought_id if path else None,
                )
                all_thoughts[thought.thought_id] = thought
                new_path = path + [thought]

                if score > best_score:
                    best_score = score
                    best_path = new_path

                next_level.append(new_path)

        # Keep only top branching_factor paths per level to control expansion
        next_level.sort(key=lambda p: p[-1].score if p else 0, reverse=True)
        current_level = next_level[: params.branching_factor]

    return best_path


async def _dfs_search(
    params: ToTReasoningInput,
    all_thoughts: dict[str, ToTThought],
    root_id: str,
) -> list[ToTThought]:
    """Depth-first: explore one branch completely. Fast but may miss better paths."""
    path: list[ToTThought] = []

    for depth in range(params.max_depth):
        path_text = _path_to_text(path)
        children = await _generate_thoughts(
            params.problem, params.context, path_text, params.branching_factor
        )
        scores = await _evaluate_thoughts_parallel(params.problem, path_text, children)

        best_idx = scores.index(max(scores))
        thought = ToTThought(
            thought_id=str(uuid.uuid4())[:8],
            depth=depth + 1,
            content=children[best_idx],
            score=scores[best_idx],
            parent_id=path[-1].thought_id if path else None,
        )
        all_thoughts[thought.thought_id] = thought
        path.append(thought)

        if await _is_terminal(params.problem, path):
            thought.is_terminal = True
            break

    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _generate_thoughts(
    problem: str,
    context: str,
    path_so_far: str,
    k: int,
) -> list[str]:
    """Ask Gemini to generate k distinct reasoning steps."""
    prompt = _THOUGHT_GEN_PROMPT.format(
        problem=problem,
        context=context[:1500] if context else "No additional context provided",
        path_so_far=path_so_far or "This is the first step.",
        k=k,
    )

    response = await generate(prompt=prompt, temperature=0.7, use_cache=False)

    thoughts: list[str] = []
    for line in response.split("\n"):
        for i in range(1, k + 1):
            prefix = f"THOUGHT_{i}:"
            if line.strip().startswith(prefix):
                thought = line.strip()[len(prefix):].strip()
                if thought:
                    thoughts.append(thought)
                break

    if not thoughts:
        thoughts = [line.strip() for line in response.strip().split("\n") if line.strip()]

    return thoughts[:k]


async def _evaluate_thoughts_parallel(
    problem: str,
    path_so_far: str,
    thoughts: list[str],
) -> list[float]:
    """Score all thoughts in parallel (concurrent Gemini calls)."""
    tasks = [
        _evaluate_single(problem, path_so_far, t)
        for t in thoughts
    ]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    result = []
    for s in scores:
        if isinstance(s, Exception):
            logger.warning("thought_eval_error", error=str(s))
            result.append(0.5)   # Default neutral score on error
        else:
            result.append(float(s))
    return result


async def _evaluate_single(
    problem: str, path_so_far: str, thought: str
) -> float:
    """Score a single thought. Returns 0.0-1.0."""
    prompt = _THOUGHT_EVAL_PROMPT.format(
        problem=problem,
        path_so_far=path_so_far or "No prior steps",
        thought=thought,
    )
    response = await generate(prompt=prompt, temperature=0.1, use_cache=True)

    try:
        return max(0.0, min(1.0, float(response.strip())))
    except ValueError:
        # Try to extract first float from response
        import re
        match = re.search(r"\d+\.\d+", response)
        return float(match.group()) if match else 0.5


async def _is_terminal(problem: str, path: list[ToTThought]) -> bool:
    """Ask the LLM if the current reasoning path has reached a conclusion."""
    if not path:
        return False

    path_text = _path_to_text(path)
    prompt = _TERMINAL_CHECK_PROMPT.format(problem=problem, path=path_text)
    response = await generate(prompt=prompt, temperature=0.0, use_cache=True)
    return response.strip().upper().startswith("YES")


def _path_to_text(path: list[ToTThought]) -> str:
    if not path:
        return "No reasoning steps yet."
    return "\n".join(
        f"Step {i+1} (score={t.score:.2f}): {t.content}"
        for i, t in enumerate(path)
    )


def _get_best_partial_path(all_thoughts: dict[str, ToTThought]) -> list[ToTThought]:
    """Fallback: return the path to the highest-scoring thought."""
    if not all_thoughts:
        return []
    best = max(all_thoughts.values(), key=lambda t: t.score)

    # Walk up the parent chain
    path = []
    current = best
    while current:
        path.append(current)
        if current.parent_id:
            current = all_thoughts.get(current.parent_id)
        else:
            break

    path.reverse()
    return path