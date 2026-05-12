"""
POST /techniques/run
Runs a prompt with one of five techniques (zero_shot, few_shot, chain_of_thought,
react, tree_of_thoughts) and returns output, token counts, and USD cost.
"""

import asyncio
import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from core.config import settings
from core.logging import get_logger

router = APIRouter(tags=["techniques"])
logger = get_logger(__name__)

_MODEL_PRICING: dict[str, dict[str, float]] = {  # USD per 1M tokens
    "gemini-2.5-flash": {"input": 0.15,  "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.10,  "output": 0.40},
}

AVAILABLE_MODELS = list(_MODEL_PRICING.keys())


class Example(BaseModel):
    input: str
    output: str


class TechniqueRequest(BaseModel):
    technique: Literal["zero_shot", "few_shot", "chain_of_thought", "react", "tree_of_thoughts"]
    model: str = "gemini-2.5-flash"
    system_prompt: str = ""
    user_query: str
    examples: list[Example] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    num_branches: int = Field(default=3, ge=2, le=5)


class BranchResult(BaseModel):
    branch: int
    reasoning: str
    answer: str


class TechniqueResponse(BaseModel):
    output: str
    prompt_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
    technique: str
    duration_ms: int
    branches: list[BranchResult] | None = None


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _MODEL_PRICING.get(model, _MODEL_PRICING["gemini-2.5-flash"])
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


async def _call(
    client: genai.Client,
    model: str,
    system: str,
    contents,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = (response.text or "").strip()
    meta = response.usage_metadata
    return text, (meta.prompt_token_count or 0), (meta.candidates_token_count or 0)


def _build_zero_shot(req: TechniqueRequest) -> tuple[str, list, str]:
    system = req.system_prompt or "You are a helpful assistant."
    contents = [req.user_query]
    display = f"[System]\n{system}\n\n[User]\n{req.user_query}"
    return system, contents, display


def _build_few_shot(req: TechniqueRequest) -> tuple[str, list, str]:
    system = req.system_prompt or "You are a helpful assistant."
    lines = []
    for i, ex in enumerate(req.examples, 1):
        lines.append(f"Example {i}:\nInput: {ex.input}\nOutput: {ex.output}")
    examples_block = "\n\n".join(lines)
    user_msg = (
        f"{examples_block}\n\nNow answer the following in the same style:\n"
        f"Input: {req.user_query}\nOutput:"
    ) if lines else req.user_query
    display = f"[System]\n{system}\n\n[User]\n{user_msg}"
    return system, [user_msg], display


def _build_cot(req: TechniqueRequest) -> tuple[str, list, str]:
    system = (req.system_prompt or "You are a helpful assistant.") + (
        "\n\nBefore giving your final answer, reason step by step. "
        "Format your response as:\nReasoning: <step-by-step thinking>\nAnswer: <final answer>"
    )
    user_msg = f"{req.user_query}\n\nThink step by step."
    display = f"[System]\n{system}\n\n[User]\n{user_msg}"
    return system, [user_msg], display


def _build_react(req: TechniqueRequest) -> tuple[str, list, str]:
    system = (req.system_prompt or "You are a helpful assistant.") + (
        "\n\nSolve the problem using the ReAct format:\n"
        "Thought: <what you are thinking>\n"
        "Action: <what action you would take or tool you would call (describe or simulate)>\n"
        "Observation: <what you would observe from that action>\n"
        "... (repeat Thought / Action / Observation as needed)\n"
        "Final Answer: <your conclusion>\n\n"
        "Work through the full reasoning before giving the Final Answer."
    )
    display = f"[System]\n{system}\n\n[User]\n{req.user_query}"
    return system, [req.user_query], display


def _parse_tot_branch(text: str) -> tuple[str, str]:
    if "Conclusion:" in text:
        parts = text.split("Conclusion:", 1)
        reasoning = parts[0].replace("Reasoning:", "").strip()
        answer = parts[1].strip()
    elif "Answer:" in text:
        parts = text.split("Answer:", 1)
        reasoning = parts[0].replace("Reasoning:", "").strip()
        answer = parts[1].strip()
    else:
        reasoning = text
        answer = text
    return reasoning, answer


@router.post("/techniques/run", response_model=TechniqueResponse)
async def run_technique(req: TechniqueRequest):
    model = req.model if req.model in _MODEL_PRICING else "gemini-2.5-flash"
    client = genai.Client(api_key=settings.gemini_api_key)

    t0 = time.perf_counter()
    total_in = total_out = 0
    branches = None

    if req.technique == "tree_of_thoughts":
        branch_system = (req.system_prompt or "You are a helpful assistant.") + (
            "\n\nGenerate a complete reasoning chain for the question below.\n"
            "Format:\nReasoning: <step-by-step thinking>\nConclusion: <your answer>"
        )
        judge_system = (
            "You are evaluating multiple candidate answers to a question. "
            "Pick the best one and return ONLY: BEST: <the full best answer text>"
        )
        prompt_display = (
            f"[Branch System]\n{branch_system}\n\n[User]\n{req.user_query}\n\n"
            f"[Judge System]\n{judge_system}\n\n"
            f"[Judge Input]\nAll {req.num_branches} branches combined"
        )

        branch_responses = await asyncio.gather(*[
            _call(client, model, branch_system, [req.user_query], req.temperature, req.max_tokens)
            for _ in range(req.num_branches)
        ])

        branches = []
        branch_texts = []
        for i, (text, in_tok, out_tok) in enumerate(branch_responses, 1):
            total_in += in_tok
            total_out += out_tok
            reasoning, answer = _parse_tot_branch(text)
            branches.append(BranchResult(branch=i, reasoning=reasoning, answer=answer))
            branch_texts.append(f"Branch {i}:\n{text}")

        judge_query = f"Question: {req.user_query}\n\n" + "\n\n---\n\n".join(branch_texts)
        judge_text, j_in, j_out = await _call(
            client, model, judge_system, [judge_query], 0.0, 512
        )
        total_in += j_in
        total_out += j_out

        if "BEST:" in judge_text:
            output = judge_text.split("BEST:", 1)[1].strip()
        else:
            output = branches[0].answer if branches else judge_text

    else:
        builder = {
            "zero_shot":         _build_zero_shot,
            "few_shot":          _build_few_shot,
            "chain_of_thought":  _build_cot,
            "react":             _build_react,
        }[req.technique]
        system, contents, prompt_display = builder(req)
        output, total_in, total_out = await _call(
            client, model, system, contents, req.temperature, req.max_tokens
        )

    duration_ms = int((time.perf_counter() - t0) * 1000)
    cost = _cost(model, total_in, total_out)

    logger.info(
        "technique_run",
        technique=req.technique,
        model=model,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=round(cost, 8),
        duration_ms=duration_ms,
    )

    return TechniqueResponse(
        output=output,
        prompt_used=prompt_display,
        input_tokens=total_in,
        output_tokens=total_out,
        total_tokens=total_in + total_out,
        cost_usd=cost,
        model=model,
        technique=req.technique,
        duration_ms=duration_ms,
        branches=branches,
    )
