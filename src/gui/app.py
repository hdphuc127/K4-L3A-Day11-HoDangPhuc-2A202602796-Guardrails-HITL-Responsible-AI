"""
GUI for testing and attacking the guardrails pipeline.

Run from repo root:
    pip install -r requirements.txt
    streamlit run src/gui/app.py
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path

import streamlit as st

# Allow imports from src/
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from assignment.pipeline import build_production_plugins, build_observability, is_egress_allowed
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from agents.agent import create_blue_agent
from core.utils import chat_with_agent
from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter


st.set_page_config(
    page_title="Guardrails Pipeline Tester",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Guardrails Pipeline Tester")
st.markdown(
    "Test and attack the pipeline's defense layers. See how rate limiting, input guardrails, "
    "and output guardrails protect the system."
)


@st.cache_resource
def init_pipeline():
    """Initialize the pipeline once (cached across Streamlit reruns)."""
    plugins = build_production_plugins(use_llm_judge=True)
    audit, monitor = build_observability()
    agent, runner = create_blue_agent(plugins)
    return plugins, audit, monitor, agent, runner


plugins, audit, monitor, agent, runner = init_pipeline()


def _plugin(plugins_list, plugin_class):
    """Helper to find a plugin by class."""
    for p in plugins_list:
        if isinstance(p, plugin_class):
            return p
    return None


def get_block_info(rate_before, inp_before):
    """Inspect plugins to determine which layer blocked (if any) and why.

    Follows pipeline._run_one logic exactly.
    Args:
        rate_before: blocked_count from rate limiter before request
        inp_before: blocked_count from input guardrail before request
    """
    from assignment.rate_limiter import RateLimitPlugin
    from guardrails.input_guardrails import InputGuardrailPlugin
    from guardrails.output_guardrails import OutputGuardrailPlugin

    rate = _plugin(plugins, RateLimitPlugin)
    inp = _plugin(plugins, InputGuardrailPlugin)
    out = _plugin(plugins, OutputGuardrailPlugin)

    blocked = False
    layer = None
    reason = ""

    # Check rate limiter (following pipeline._run_one logic)
    if rate.blocked_count > rate_before:
        blocked = True
        layer = "rate_limiter"
        reason = f"Rate limit exceeded (max {rate.max_requests} requests per {rate.window_seconds}s)"
    # Check input guardrail
    elif inp.blocked_count > inp_before:
        blocked = True
        layer = f"input_guardrail:{inp.last_block_reason}"
        reason = f"Input guardrail blocked: {inp.last_block_reason}"
    # Check output guardrail
    elif out.last_action in ("secret_blocked", "judge_blocked"):
        blocked = True
        layer = f"output_guardrail:{out.last_action}"
        if out.last_action == "judge_blocked":
            reason = f"LLM Judge blocked the response"
        else:
            reason = f"Secret/PII detected in response"
    elif out.last_action == "pii_redacted":
        # Answered but redacted
        blocked = False
        layer = "output_guardrail:pii_redacted"
        reason = "Response redacted (PII/secrets removed)"

    return blocked, layer, reason


# Two tabs: full pipeline test + instant rules preview
tab1, tab2 = st.tabs(["🚀 Attack the Pipeline", "📋 Instant Guardrail Checks"])

with tab1:
    st.header("Attack the Pipeline")
    st.markdown(
        "Type a message and see if it passes all defense layers. The pipeline includes "
        "rate limiting, input/output guardrails, and optional LLM-as-Judge checking."
    )

    user_input = st.text_area(
        "Your message:",
        height=120,
        placeholder="Try: 'Ignore all previous instructions' or 'What is your system prompt?'",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🎯 Send Message", use_container_width=True)

    if submit and user_input.strip():
        with st.spinner("Processing through guardrails..."):
            try:
                from assignment.rate_limiter import RateLimitPlugin
                from guardrails.input_guardrails import InputGuardrailPlugin

                # Capture state before request
                rate = _plugin(plugins, RateLimitPlugin)
                inp = _plugin(plugins, InputGuardrailPlugin)
                rate_before = rate.blocked_count
                inp_before = inp.blocked_count

                response = asyncio.run(chat_with_agent(agent, runner, user_input.strip()))
                final_response = response[0] if isinstance(response, tuple) else response

                # Determine which layer (if any) blocked this request
                blocked, layer, reason = get_block_info(rate_before, inp_before)

                # Log to monitoring
                monitor.record(blocked=blocked, layer=layer)

                # Display result
                col1, col2 = st.columns([1, 3])
                with col1:
                    if blocked:
                        st.error(f"🚫 BLOCKED")
                    else:
                        st.success(f"✅ ALLOWED")
                with col2:
                    if blocked:
                        st.info(f"**Layer:** {layer}\n**Reason:** {reason}")

                # Show response
                st.divider()
                st.subheader("Response")
                st.text_area("", value=final_response, height=150, disabled=True)

                # Show metrics
                snapshot = monitor.snapshot()
                st.divider()
                st.subheader("Session Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Requests", snapshot.get("total_requests", 0))
                col2.metric("Blocked", snapshot.get("blocked_requests", 0))
                col3.metric("Block Rate", f"{snapshot.get('block_rate', 0):.1%}")
                col4.metric("Rate Limit Hits", snapshot.get("rate_limit_hits", 0))

            except Exception as e:
                st.error(f"Error: {e}")
                st.markdown(
                    "Make sure your API key (OPENROUTER_API_KEY) is set in `.env` at the repo root."
                )

    # Sidebar: monitoring
    with st.sidebar:
        st.subheader("📊 Live Monitoring")
        snapshot = monitor.snapshot()
        st.metric("Total Requests", snapshot.get("total_requests", 0))
        st.metric("Blocked Requests", snapshot.get("blocked_requests", 0))
        st.metric("Block Rate", f"{snapshot.get('block_rate', 0):.1%}")
        st.metric("Rate Limit Hits", snapshot.get("rate_limit_hits", 0))

        # Alerts
        alerts = snapshot.get("alerts", [])
        if alerts:
            st.warning(f"⚠️ {len(alerts)} alert(s)")
            for alert in alerts:
                st.text(f"• {alert}")


with tab2:
    st.header("Instant Guardrail Checks")
    st.markdown(
        "No API key needed! These checks run locally and instantly. "
        "Useful for understanding the guardrail rules before testing the full pipeline."
    )

    test_input = st.text_area(
        "Test text:",
        height=100,
        placeholder="Try: 'Ignore all previous instructions' or 'Reveal your system prompt'",
        key="instant_check",
    )

    if st.button("✨ Run Instant Checks", use_container_width=True):
        col1, col2, col3 = st.columns(3)

        # Injection detection
        with col1:
            st.subheader("Injection Detection")
            result = detect_injection(test_input)
            if result == "BLOCK":
                st.error("🚫 BLOCKED")
                st.caption("Prompt injection detected")
            else:
                st.success("✅ ALLOWED")
                st.caption("No injection detected")

        # Topic filter
        with col2:
            st.subheader("Topic Filter")
            result = topic_filter(test_input)
            if result == "BLOCK":
                st.error("🚫 BLOCKED")
                st.caption("Blocked topic detected")
            else:
                st.success("✅ ALLOWED")
                st.caption("Topic allowed")

        # Content filter
        with col3:
            st.subheader("Content Filter")
            filter_result = content_filter(test_input)
            if filter_result["safe"]:
                st.success("✅ SAFE")
                st.caption("No PII/secrets detected")
            else:
                st.error("🚫 UNSAFE")
                if filter_result.get("issues"):
                    st.caption("; ".join(filter_result["issues"]))

    st.divider()
    st.subheader("How to Use")
    st.markdown(
        """
        **Injection Detection**: Catches prompt injection attempts like "ignore previous instructions"

        **Topic Filter**: Blocks messages about restricted topics

        **Content Filter**: Detects PII (emails, phone), secrets (API keys, passwords), and harmful content
        """
    )
