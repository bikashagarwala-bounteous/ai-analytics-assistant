"""
Prompt Lab — test, compare, and optimize prompt templates.

Four panels:
  1. Single test         — run a query through the analytics pipeline
  2. A/B compare         — run the same query twice and diff results
  3. Optimize            — generate improved prompt variants
  4. Techniques Playground — experiment with zero-shot, few-shot, CoT, ReAct, ToT
"""

import time

import streamlit as st

from components.api_client import post, get


def render():
    st.title("Prompt Lab")
    st.caption("Test and optimize the prompts used by the analytics agents.")

    tab_test, tab_ab, tab_optimize, tab_tech = st.tabs(
        ["Single Test", "A/B Compare", "Optimize", "Techniques Playground"]
    )

    with tab_test:
        _render_single_test()

    with tab_ab:
        _render_ab_compare()

    with tab_optimize:
        _render_optimizer()

    with tab_tech:
        _render_techniques()


# ── Single test ───────────────────────────────────────────────────────────────

def _render_single_test():
    st.subheader("Test a Query")
    st.markdown("Run any query through the analytics pipeline and inspect the raw output.")

    query = st.text_area(
        "Query",
        value="Why did engagement drop last week?",
        height=80,
        key="single_query",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Analysis", type="primary", use_container_width=True, key="run_single"):
            if not query.strip():
                st.warning("Enter a query first.")
                return
            _run_and_display(query)

    with col2:
        if st.button("Clear", use_container_width=True, key="clear_single"):
            for key in ["single_result", "single_error"]:
                st.session_state.pop(key, None)
            st.rerun()

    if result := st.session_state.get("single_result"):
        _display_analysis_result(result)

    if error := st.session_state.get("single_error"):
        st.error(error)


def _run_and_display(query: str):
    with st.spinner("Running pipeline..."):
        t0 = time.perf_counter()
        result = post("/analyze", {"query": query})
        duration = int((time.perf_counter() - t0) * 1000)

    if result:
        result["_client_duration_ms"] = duration
        st.session_state["single_result"] = result
        st.session_state.pop("single_error", None)
    else:
        st.session_state["single_error"] = "Analysis failed. Check backend logs."
    st.rerun()


def _display_analysis_result(result: dict):
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        font-size: 12px;
    }

    div[data-testid="stMetric"] label {
        font-size: 11px !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Status", result.get("status", "—"))
    c2.metric("Confidence", f"{result.get('confidence', 0):.0%}")
    c3.metric("Tool Calls", len(result.get("tool_calls", [])))
    c4.metric("Duration", f"{result.get('_client_duration_ms', 0)}ms")

    st.subheader("Summary")
    st.markdown(result.get("summary", "—"))

    with st.expander("Findings"):
        for f in result.get("findings", []):
            st.markdown(f"- {f}")

    with st.expander("Root Causes"):
        for r in result.get("root_causes", []):
            st.markdown(f"- {r}")

    with st.expander("Recommended Actions"):
        for a in result.get("recommended_actions", []):
            st.markdown(f"- {a}")

    with st.expander("Tool Calls Detail"):
        for tc in result.get("tool_calls", []):
            status = "✓" if tc.get("success") else "✗"
            st.markdown(
                f"{status} **{tc['tool_name']}** — "
                f"{tc.get('duration_ms', 0):.0f}ms"
                + (f" — `{tc['error']}`" if tc.get("error") else "")
            )


# ── A/B compare ───────────────────────────────────────────────────────────────

def _render_ab_compare():
    st.subheader("A/B Compare")
    st.markdown("Run the same query twice and compare outputs — useful after prompt changes.")

    query = st.text_area(
        "Query (same for both runs)",
        value="What are the top failure intents?",
        height=80,
        key="ab_query",
    )

    if st.button("Run Both", type="primary", use_container_width=True, key="run_ab"):
        if not query.strip():
            st.warning("Enter a query first.")
            return

        results = []
        with st.spinner("Running two parallel analyses..."):
            import asyncio
            import httpx
            import json
            import os

            backend = os.getenv("BACKEND_URL", "http://localhost:8080")

            async def _run_one():
                async with httpx.AsyncClient(timeout=120) as c:
                    r = await c.post(f"{backend}/analyze", json={"query": query})
                    return r.json()

            async def _run_both():
                return await asyncio.gather(_run_one(), _run_one())

            try:
                results = asyncio.run(_run_both())
            except Exception as e:
                st.error(f"Failed: {e}")
                return

        st.session_state["ab_results"] = results
        st.rerun()

    if ab := st.session_state.get("ab_results"):
        col_a, col_b = st.columns(2)
        for col, result, label in zip([col_a, col_b], ab, ["Run A", "Run B"]):
            with col:
                st.subheader(label)
                if result:
                    _display_analysis_result(result)
                else:
                    st.error("Run failed")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def _render_optimizer():
    st.subheader("Prompt Optimizer")
    st.markdown(
        "Analyse a prompt's historical performance and generate improved variants. "
        "Variants are saved to the database and can be activated."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        prompt_key = st.selectbox(
            "Prompt to optimize",
            ["analyst_system", "summary_system"],
            key="opt_prompt_key",
        )
    with col2:
        goal = st.selectbox(
            "Goal",
            ["user_satisfaction", "accuracy", "conciseness", "failure_reduction"],
            key="opt_goal",
        )

    current_template = st.text_area(
        "Current template (paste the prompt you want to improve)",
        height=150,
        key="opt_template",
        placeholder="Paste the current prompt template here...",
    )

    n_variants = st.slider("Variants to generate", 1, 5, 3, key="opt_n")

    if st.button("Generate Variants", type="primary", key="run_opt"):
        if not current_template.strip():
            st.warning("Paste a prompt template to optimize.")
            return

        with st.spinner("Analysing performance and generating variants..."):
            result = post("/optimize", {
                "prompt_key": prompt_key,
                "current_template": current_template,
                "optimization_goal": goal,
                "n_variants": n_variants,
                "use_feedback_data": True,
            })

        if result and not result.get("error"):
            st.success(f"Generated {len(result.get('variants', []))} variants. See results below.")
            st.session_state["opt_result"] = result
            st.rerun()
        else:
            st.error(result.get("error") if result else "Optimization failed.")

    if opt := st.session_state.get("opt_result"):
        st.divider()
        st.subheader("Optimization Result")

        if opt.get("optimization_rationale"):
            st.markdown(f"**Rationale:** {opt['optimization_rationale']}")

        if opt.get("expected_improvement"):
            st.caption(opt["expected_improvement"])

        for v in opt.get("variants", []):
            with st.expander(f"Variant `{v.get('variant_id', '')}` — {v.get('description', '')}"):
                st.code(v.get("template", ""), language="text")

        if opt.get("recommended_variant_id"):
            st.info(f"Recommended variant ID: `{opt['recommended_variant_id']}`")

    # ── Performance history ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Recent Run History")
    runs = get("/analyze/runs", params={"limit": 10})
    if runs and runs.get("runs"):
        for run in runs["runs"]:
            status_icon = "✓" if run["status"] == "completed" else "✗"
            with st.container(border=True):
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(f"{status_icon} {run['query']}")
                c2.caption(f"{run.get('duration_ms', '—')}ms")
                c3.caption(run.get("created_at", "")[:10])
    else:
        st.info("No runs yet.")


# ── Techniques Playground ─────────────────────────────────────────────────────

_TECHNIQUE_INFO = {
    "zero_shot": {
        "label": "Zero-Shot",
        "desc": "Ask the model directly with only a system prompt — no examples provided.",
    },
    "few_shot": {
        "label": "Few-Shot",
        "desc": "Provide input/output examples so the model learns the pattern before answering.",
    },
    "chain_of_thought": {
        "label": "Chain-of-Thought (CoT)",
        "desc": "Force step-by-step reasoning before the final answer to improve accuracy on complex tasks.",
    },
    "react": {
        "label": "ReAct",
        "desc": "Interleave Thought → Action → Observation cycles. Actions are simulated (no live tools).",
    },
    "tree_of_thoughts": {
        "label": "Tree of Thoughts (ToT)",
        "desc": "Generate N independent reasoning branches in parallel, then a judge picks the best one.",
    },
}

_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

_MODEL_PRICING = {
    "gemini-2.5-flash": {"input": 0.15,  "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10,  "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
}


def _render_techniques():
    st.subheader("Techniques Playground")
    st.markdown(
        "Experiment with different prompting strategies. "
        "Each run shows token usage and cost so you can compare approaches."
    )

    # ── Config row ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        technique = st.selectbox(
            "Technique",
            list(_TECHNIQUE_INFO.keys()),
            format_func=lambda k: _TECHNIQUE_INFO[k]["label"],
            key="tech_technique",
        )
    with c2:
        model = st.selectbox("Model", _MODELS, key="tech_model")
    with c3:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, key="tech_temp")

    st.caption(_TECHNIQUE_INFO[technique]["desc"])
    st.divider()

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=90,
        key="tech_system",
        placeholder="Instructions that set the model's role and behaviour…",
    )

    # ── Few-shot examples ─────────────────────────────────────────────────────
    if technique == "few_shot":
        st.markdown("**Examples** — add input/output pairs to demonstrate the pattern")
        if "tech_examples" not in st.session_state:
            st.session_state.tech_examples = [{"input": "", "output": ""}]

        for i, ex in enumerate(st.session_state.tech_examples):
            ecol1, ecol2, ecol3 = st.columns([5, 5, 1])
            ex["input"] = ecol1.text_input(
                f"Input {i + 1}", value=ex["input"], key=f"tech_ex_in_{i}"
            )
            ex["output"] = ecol2.text_input(
                f"Output {i + 1}", value=ex["output"], key=f"tech_ex_out_{i}"
            )
            if ecol3.button("✕", key=f"tech_ex_del_{i}") and len(st.session_state.tech_examples) > 1:
                st.session_state.tech_examples.pop(i)
                st.rerun()

        if st.button("+ Add Example", key="tech_add_ex"):
            st.session_state.tech_examples.append({"input": "", "output": ""})
            st.rerun()
    else:
        st.session_state.pop("tech_examples", None)

    # ── ToT branches ──────────────────────────────────────────────────────────
    num_branches = 3
    if technique == "tree_of_thoughts":
        num_branches = st.slider("Reasoning branches", 2, 5, 3, key="tech_branches")
        st.caption(
            f"The model will generate {num_branches} independent reasoning paths "
            "and a judge will pick the best answer. Uses more tokens."
        )

    # ── Query + token budget ──────────────────────────────────────────────────
    user_query = st.text_area(
        "User Query",
        height=100,
        key="tech_query",
        placeholder="Enter your question or task here…",
    )

    max_tokens = st.slider("Max output tokens", 64, 4096, 1024, 64, key="tech_max_tokens")

    # ── Action buttons ────────────────────────────────────────────────────────
    run_col, clear_col, _ = st.columns([1, 1, 4])
    run = run_col.button("Run", type="primary", use_container_width=True, key="tech_run")
    if clear_col.button("Clear", use_container_width=True, key="tech_clear"):
        for k in ["tech_result", "tech_error"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run:
        if not user_query.strip():
            st.warning("Enter a query first.")
        else:
            examples = []
            if technique == "few_shot":
                examples = [
                    {"input": e["input"], "output": e["output"]}
                    for e in st.session_state.get("tech_examples", [])
                    if e["input"].strip()
                ]

            payload = {
                "technique": technique,
                "model": model,
                "system_prompt": system_prompt,
                "user_query": user_query,
                "examples": examples,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_branches": num_branches,
            }

            with st.spinner("Running…"):
                result = post("/techniques/run", payload)

            if result:
                st.session_state["tech_result"] = result
                st.session_state.pop("tech_error", None)
            else:
                st.session_state["tech_error"] = "Request failed — check backend logs."
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    if error := st.session_state.get("tech_error"):
        st.error(error)

    if result := st.session_state.get("tech_result"):
        st.divider()
        st.subheader("Result")

        pricing = _MODEL_PRICING.get(result["model"], {"input": 0, "output": 0})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Input Tokens",  f"{result['input_tokens']:,}")
        m2.metric("Output Tokens", f"{result['output_tokens']:,}")
        m3.metric("Total Tokens",  f"{result['total_tokens']:,}")
        m4.metric(
            "Cost (USD)",
            f"${result['cost_usd']:.6f}",
            help=(
                f"{result['model']} pricing: "
                f"${pricing['input']}/1M input · ${pricing['output']}/1M output"
            ),
        )
        m5.metric("Duration", f"{result['duration_ms']}ms")

        st.markdown("**Output**")
        st.markdown(result["output"])

        if result.get("branches"):
            with st.expander(f"All {len(result['branches'])} reasoning branches"):
                for b in result["branches"]:
                    st.markdown(f"**Branch {b['branch']}**")
                    if b["reasoning"] != b["answer"]:
                        st.caption("Reasoning")
                        st.markdown(b["reasoning"])
                    st.caption("Answer")
                    st.markdown(b["answer"])
                    st.divider()

        with st.expander("Prompt sent to model"):
            st.code(result["prompt_used"], language="markdown")


render()