from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _positive_timeout_seconds(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


API_GET_TIMEOUT_SECONDS = _positive_timeout_seconds(
    "STREAMLIT_API_GET_TIMEOUT_SECONDS", 30.0
)
API_POST_TIMEOUT_SECONDS = _positive_timeout_seconds(
    "STREAMLIT_API_TIMEOUT_SECONDS", 360.0
)
API_GET_TIMEOUT = httpx.Timeout(API_GET_TIMEOUT_SECONDS, connect=5.0)
API_POST_TIMEOUT = httpx.Timeout(API_POST_TIMEOUT_SECONDS, connect=10.0)

st.set_page_config(
    page_title="Retention Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

            :root {
                --jdl-bg-1: #f4f8ff;
                --jdl-bg-2: #edf6f2;
                --jdl-bg-3: #fff8ef;
                --jdl-ink-900: #0d1b2f;
                --jdl-ink-700: #2b3f5d;
                --jdl-ink-600: #425979;
                --jdl-accent: #0f5cc0;
                --jdl-accent-2: #18a999;
                --jdl-card-border: rgba(152, 170, 197, 0.35);
                --jdl-card-bg: rgba(255, 255, 255, 0.78);
                --jdl-shadow: 0 14px 40px rgba(20, 40, 74, 0.12);
            }

            html, body, [class*="css"], [data-testid="stAppViewContainer"] {
                font-family: 'Manrope', sans-serif;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 12% 6%, rgba(24, 169, 153, 0.22), transparent 36%),
                    radial-gradient(circle at 82% 0%, rgba(15, 92, 192, 0.20), transparent 38%),
                    linear-gradient(135deg, var(--jdl-bg-1) 0%, var(--jdl-bg-2) 48%, var(--jdl-bg-3) 100%);
                color: var(--jdl-ink-900);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f2a4a 0%, #143b69 58%, #1a4c81 100%);
            }

            [data-testid="stSidebar"] * {
                color: #eff6ff;
            }

            .block-container {
                padding-top: 1.55rem;
                padding-bottom: 3rem;
                max-width: 1260px;
            }

            h1, h2, h3, h4 {
                font-family: 'Sora', sans-serif;
                letter-spacing: -0.02em;
                color: var(--jdl-ink-900);
            }

            [data-testid="stMetric"] {
                border: 1px solid var(--jdl-card-border);
                background: linear-gradient(145deg, rgba(255, 255, 255, 0.85) 0%, rgba(247, 251, 255, 0.78) 100%);
                border-radius: 14px;
                padding: 10px 12px;
                box-shadow: var(--jdl-shadow);
            }

            [data-testid="stMetricValue"] {
                font-size: 1.7rem;
                font-weight: 800;
                color: #11325b;
            }

            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border: 1px solid var(--jdl-card-border);
                border-radius: 14px;
                overflow: hidden;
                box-shadow: var(--jdl-shadow);
                background: var(--jdl-card-bg);
            }

            div.stButton > button, div.stDownloadButton > button {
                border-radius: 12px;
                border: 0;
                font-weight: 700;
                color: #ffffff;
                background: linear-gradient(135deg, #0f5cc0 0%, #147dcf 46%, #18a999 100%);
                box-shadow: 0 10px 22px rgba(16, 84, 170, 0.28);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            div.stButton > button:hover, div.stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 14px 26px rgba(16, 84, 170, 0.35);
            }

            [data-testid="stExpander"] {
                border-radius: 12px;
                border: 1px solid var(--jdl-card-border);
                background: rgba(255, 255, 255, 0.72);
            }

            [data-baseweb="tab-list"] {
                gap: 6px;
            }

            [data-baseweb="tab"] {
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.6);
            }

            .jdl-logo {
                display: flex;
                align-items: center;
                gap: 12px;
                margin: 0 0 10px 0;
                padding: 10px 11px;
                border: 1px solid rgba(199, 221, 255, 0.4);
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.24) 0%, rgba(172, 216, 255, 0.20) 100%);
                backdrop-filter: blur(2px);
            }

            .jdl-mark {
                width: 40px;
                height: 40px;
                border-radius: 11px;
                background: linear-gradient(160deg, #0b2440 0%, #1f5f9f 100%);
                color: #ffffff;
                font-weight: 800;
                font-size: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                letter-spacing: 0.45px;
                box-shadow: 0 8px 18px rgba(14, 39, 71, 0.28);
            }

            .jdl-text {
                line-height: 1.05;
            }

            .jdl-name {
                font-size: 1.03rem;
                font-weight: 800;
                letter-spacing: 0.2px;
            }

            .jdl-tag {
                font-size: 0.72rem;
                color: #d7e8ff;
                text-transform: uppercase;
                letter-spacing: 0.88px;
                margin-top: 4px;
            }

            .jdl-top-right {
                position: fixed;
                top: 62px;
                right: 22px;
                z-index: 700;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 5px 10px;
                border-radius: 999px;
                border: 1px solid rgba(184, 210, 245, 0.85);
                background: rgba(255, 255, 255, 0.7);
                box-shadow: 0 8px 20px rgba(14, 39, 71, 0.16);
                backdrop-filter: blur(6px);
            }

            .jdl-top-right-mark {
                width: 22px;
                height: 22px;
                border-radius: 6px;
                background: linear-gradient(160deg, #0f2a4a 0%, #1b4f8f 100%);
                color: #ffffff;
                font-weight: 800;
                font-size: 10px;
                letter-spacing: 0.2px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .jdl-top-right-text {
                color: #1c467b;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.25px;
            }

            .risk-high {color: #b42318; font-weight: 700;}
            .risk-medium {color: #b54708; font-weight: 700;}
            .risk-low {color: #027a48; font-weight: 700;}
            .evidence-card {border: 1px solid #d0d5dd; border-radius: 10px; padding: 12px; margin-bottom: 8px;}
            .small-muted {color: #667085; font-size: 0.86rem;}

            @media (max-width: 960px) {
                .jdl-top-right {
                    display: none;
                }
                .block-container {
                    padding-top: 1rem;
                }
            }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict[str, Any] | None = None) -> Any:
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=API_GET_TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=API_POST_TIMEOUT)
    response.raise_for_status()
    api_get.clear()
    return response.json()


def api_status_error_message(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code
    detail: Any = None
    try:
        body = exc.response.json()
        detail = body.get("detail") if isinstance(body, dict) else body
    except ValueError:
        detail = exc.response.text.strip()
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, ensure_ascii=False)
    return (
        f"The API returned HTTP {status_code}: {detail}"
        if detail
        else f"The API returned HTTP {status_code}. Check the API terminal for details."
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json_file(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def risk_label(value: str) -> str:
    return {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}.get(value, value.title())


def connection_guard() -> bool:
    try:
        health = api_get("/health")
        st.sidebar.success(
            f"API {health['status']} · {health['agent_provider']} · {health.get('model', 'n/a')}",
            icon="✅",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Cannot connect to the FastAPI service. Start it with "
            "`uvicorn app.api.main:app --reload` and refresh this page."
        )
        st.exception(exc)
        return False


st.sidebar.markdown(
        """
        <div class="jdl-logo">
            <div class="jdl-mark">JDL</div>
            <div class="jdl-text">
                <div class="jdl-name">John Data Lab</div>
                <div class="jdl-tag">Enterprise Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)
st.sidebar.title("Retention Intelligence")
st.sidebar.caption("Enterprise AI decision-support console")
page = st.sidebar.radio(
    "Workspace",
    [
        "Risk Dashboard",
        "Retention Copilot",
        "Customer Analysis",
        "Agent Evaluation",
        "System Trace",
    ],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption(f"API: {API_URL}")

if not connection_guard():
    st.stop()

st.markdown(
        """
        <div class="jdl-top-right">
            <div class="jdl-top-right-mark">JDL</div>
            <div class="jdl-top-right-text">John Data Lab</div>
        </div>
        """,
        unsafe_allow_html=True,
)


if page == "Retention Copilot":
    title_left, title_right = st.columns([5, 1])
    with title_left:
        st.title("Retention Copilot")
        st.caption(
            "Ask portfolio or customer questions. The Agent selects read-only tools and grounds its answer in current data."
        )
    with title_right:
        clear_chat = st.button("Clear chat", use_container_width=True)

    if clear_chat or "retention_chat_messages" not in st.session_state:
        st.session_state.retention_chat_messages = []

    st.info(
        "This Copilot can query customer facts, ML risk, portfolio rankings, Memory, approved playbooks, "
        "Skills, interventions and ROI scenarios. It cannot execute arbitrary SQL or external actions."
    )

    suggestion_columns = st.columns(3)
    suggested_prompt = None
    with suggestion_columns[0]:
        if st.button("Top 10 risk customers", use_container_width=True):
            suggested_prompt = "帮我选择风险最高的10个客户，并说明排序依据"
    with suggestion_columns[1]:
        if st.button("Why not discount?", use_container_width=True):
            suggested_prompt = "为什么02客户不应该优先使用打折的方式？请查询相关数据和历史信息"
    with suggestion_columns[2]:
        if st.button("Portfolio summary", use_container_width=True):
            suggested_prompt = "总结当前客户组合的风险，并告诉我应该先关注什么"

    for message in st.session_state.retention_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("meta"):
                meta = message["meta"]
                tools = ", ".join(meta.get("tools_used", [])) or "none"
                fallback = " · fallback" if meta.get("fallback_used") else ""
                st.caption(
                    f"Provider: {meta.get('provider', 'unknown')} · Model: {meta.get('model', 'unknown')}"
                    f"{fallback} · Tools: {tools}"
                )

    typed_prompt = st.chat_input(
        "例如：为什么02客户不能优先打折？或者帮我找风险最高的10个客户"
    )
    prompt = suggested_prompt or typed_prompt
    if prompt:
        prior_history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.retention_chat_messages[-12:]
        ]
        st.session_state.retention_chat_messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(
                "Selecting tools and checking evidence... "
                "Local Ollama may take several minutes."
            ):
                try:
                    payload = api_post(
                        "/v1/chat",
                        {
                            "message": prompt,
                            "session_id": "streamlit-retention-copilot",
                            "history": prior_history,
                        },
                    )["data"]
                    st.markdown(payload["answer"])
                    tools = ", ".join(payload.get("tools_used", [])) or "none"
                    fallback = " · fallback" if payload.get("fallback_used") else ""
                    st.caption(
                        f"Provider: {payload['provider']} · Model: {payload['model']}"
                        f"{fallback} · Tools: {tools}"
                    )
                    st.session_state.retention_chat_messages.append(
                        {
                            "role": "assistant",
                            "content": payload["answer"],
                            "meta": payload,
                        }
                    )
                except httpx.ReadTimeout:
                    st.error(
                        "The local Agent exceeded the UI wait limit "
                        f"({API_POST_TIMEOUT_SECONDS:.0f} seconds). "
                        "The backend may still finish the request; check System Trace before retrying."
                    )
                except httpx.HTTPStatusError as exc:
                    st.error(api_status_error_message(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "The Retention Copilot could not reach the API. "
                        f"{type(exc).__name__}: {exc}"
                    )


elif page == "Risk Dashboard":
    st.title("Risk Dashboard")
    st.caption("Prioritize accounts by churn risk, renewal urgency, and revenue exposure.")

    payload = api_get("/v1/operations/dashboard", {"limit": 1000})
    summary = payload["summary"]
    customers = pd.DataFrame(payload["customers"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("High-risk customers", summary["high_risk_customers"])
    c2.metric("Revenue at risk", money(summary["revenue_at_risk"]))
    c3.metric("Renewal ≤ 30 days", summary["renewal_within_30_days"])
    c4.metric("Pending approvals", summary["pending_approvals"])
    c5.metric("Total customers", summary["total_customers"])

    st.subheader("Customer portfolio")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    with f1:
        selected_risks = st.multiselect(
            "Risk band", ["high", "medium", "low"], default=["high", "medium"]
        )
    with f2:
        segments = sorted(customers["segment"].dropna().unique().tolist())
        selected_segments = st.multiselect("Segment", segments, default=segments)
    with f3:
        max_renewal = st.slider("Renewal within days", 0, 365, 120)
    with f4:
        search = st.text_input("Search customer", placeholder="CUST-0283")

    filtered = customers[
        customers["risk_band"].isin(selected_risks)
        & customers["segment"].isin(selected_segments)
        & (customers["days_to_renewal"] <= max_renewal)
    ].copy()
    if search:
        filtered = filtered[filtered["customer_id"].str.contains(search, case=False, na=False)]

    chart_left, chart_right = st.columns([1.15, 1])
    with chart_left:
        st.markdown("#### Highest-risk accounts")
        chart_data = (
            filtered.nlargest(12, "churn_probability")
            .set_index("customer_id")[["churn_probability"]]
            .rename(columns={"churn_probability": "Churn probability"})
        )
        st.bar_chart(chart_data, horizontal=True)
    with chart_right:
        st.markdown("#### Revenue exposure by segment")
        segment_data = (
            filtered.groupby("segment", as_index=True)["revenue_at_risk"].sum().sort_values()
        )
        st.bar_chart(segment_data, horizontal=True)

    display = filtered.sort_values(["churn_probability", "revenue_at_risk"], ascending=False).copy()
    display["Churn risk"] = display["churn_probability"].map(lambda x: f"{x:.1%}")
    display["Risk band"] = display["risk_band"].map(risk_label)
    display["Monthly revenue"] = display["monthly_revenue"].map(money)
    display["Revenue at risk"] = display["revenue_at_risk"].map(money)
    display["Usage change"] = display["usage_change_pct"].map(lambda x: f"{x:.1f}%")
    display = display.rename(
        columns={
            "customer_id": "Customer",
            "segment": "Segment",
            "region": "Region",
            "days_to_renewal": "Renewal days",
            "support_tickets_90d": "Support tickets",
            "payment_failures_90d": "Payment failures",
        }
    )
    st.dataframe(
        display[
            [
                "Customer",
                "Risk band",
                "Churn risk",
                "Segment",
                "Region",
                "Revenue at risk",
                "Renewal days",
                "Usage change",
                "Support tickets",
                "Payment failures",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=430,
    )
    st.caption(f"Showing {len(display):,} customers after filters.")

    st.subheader("Model comparison snapshot")
    root = _project_root()
    leaderboard_path = root / "artifacts" / "models" / "tuning" / "grid_search" / "leaderboard.json"
    leaderboard_data = _read_json_file(leaderboard_path)
    if not leaderboard_data:
        st.info("No tuning leaderboard found yet.")
    else:
        snapshot_rows: list[dict[str, Any]] = []
        for item in leaderboard_data:
            snapshot_rows.append(
                {
                    "Model": item.get("model_type", "n/a"),
                    "Metric": item.get("score_metric", "n/a"),
                    "Score": item.get("score", "n/a"),
                }
            )
        st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)
        st.caption("Full details (run id + best params) are available in Workspace → Agent Evaluation.")


elif page == "Customer Analysis":
    st.title("Customer Analysis")
    st.caption("Combine ML prediction, customer memory, enterprise knowledge, and agent review.")

    dashboard = api_get("/v1/operations/dashboard", {"limit": 1000})
    customer_options = [item["customer_id"] for item in dashboard["customers"]]
    default_customer = customer_options[0] if customer_options else "CUST-0001"

    top1, top2, top3 = st.columns([1.3, 1, 1])
    with top1:
        customer_id = st.selectbox("Customer", customer_options, index=0)
    with top2:
        session_id = st.text_input("Session ID", value="streamlit-demo")
    with top3:
        create_ticket = st.checkbox("Create approval ticket", value=False)

    if "latest_analysis" not in st.session_state:
        st.session_state.latest_analysis = None

    if st.button("Run enterprise analysis", type="primary", use_container_width=True):
        with st.spinner(
            "Running ML, memory retrieval, RAG, recommendation, and review... "
            "Local Ollama may take 2–5 minutes."
        ):
            try:
                response = api_post(
                    "/v1/analysis",
                    {
                        "customer_id": customer_id,
                        "session_id": session_id,
                        "create_ticket": create_ticket,
                    },
                )
                st.session_state.latest_analysis = response["data"]
            except httpx.ReadTimeout:
                st.error(
                    "The local Agent exceeded the UI wait limit "
                    f"({API_POST_TIMEOUT_SECONDS:.0f} seconds). "
                    "The backend may still complete and save the analysis. "
                    "Check System Trace before running it again."
                )
            except httpx.HTTPStatusError as exc:
                st.error(api_status_error_message(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "The analysis could not reach the API. "
                    f"{type(exc).__name__}: {exc}"
                )

    data = st.session_state.latest_analysis
    if data:
        customer = data["customer"]
        prediction = data["prediction"]
        recommendation = data["recommendation"]
        review = data["review"]

        st.divider()
        st.subheader(f"{customer['customer_id']} · {customer['segment']} · {customer['region']}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Churn probability", f"{prediction['churn_probability']:.1%}")
        m2.metric("Risk band", risk_label(prediction["risk_band"]))
        m3.metric("Revenue at risk", money(customer["revenue_at_risk"]))
        m4.metric("Renewal", f"{customer['days_to_renewal']} days")
        m5.metric("Workflow latency", f"{data['latency_ms']:.0f} ms")

        if data.get("fallback_used", False):
            st.warning(
                f"Deterministic fallback was used: {data.get('fallback_reason') or 'the requested local LLM path was unavailable.'}",
                icon="⚠️",
            )
        elif not data.get("provider_available", data.get("ollama_available", True)):
            st.warning(
                "The configured model provider is currently unavailable, so the interactive LLM recommendation path is degraded.",
                icon="⚠️",
            )

        st.caption(
            f"Requested provider: {data.get('requested_provider', 'n/a')} · Active provider: {data.get('provider', 'n/a')} · Generation mode: {data.get('generation_mode', 'n/a')} · Model: {data.get('model', 'n/a')} · LLM latency: {data.get('llm_latency_ms', 'n/a')}"
        )

        left, center, right = st.columns([1.05, 1.35, 1.05], gap="large")

        with left:
            st.markdown("### Why this customer is at risk")
            driver_rows = []
            for driver in prediction["top_drivers"]:
                driver_rows.append(
                    {
                        "Driver": driver["feature"].replace("_", " ").title(),
                        "Customer": driver["customer_value"],
                        "Baseline": driver["baseline_value"],
                        "Impact": driver["probability_impact"],
                        "Direction": driver["direction"],
                    }
                )
                icon = "↑" if driver["direction"] == "risk" else "↓"
                st.markdown(
                    f"**{icon} {driver['feature'].replace('_', ' ').title()}**  \n"
                    f"{driver['explanation']}"
                )
            if driver_rows:
                driver_df = pd.DataFrame(driver_rows).set_index("Driver")[["Impact"]]
                st.bar_chart(driver_df, horizontal=True)

            st.markdown("### Customer memory")
            if data["memories_used"]:
                for memory in data["memories_used"]:
                    st.info(memory["content"], icon="🧠")
            else:
                st.caption(
                    "No long-term memory existed before this analysis. The result is now recorded."
                )

            with st.expander("Customer profile"):
                profile_table = pd.DataFrame(
                    [
                        {"Metric": key.replace("_", " ").title(), "Value": value}
                        for key, value in customer.items()
                    ]
                )
                st.dataframe(profile_table, hide_index=True, use_container_width=True)

        with center:
            st.markdown("### Recommended retention plan")
            st.success(recommendation["executive_summary"], icon="🎯")
            for action in recommendation["actions"]:
                approval = " · Approval required" if action["approval_required"] else ""
                st.markdown(
                    f"**{action['priority']}. {action['action']}**  \n"
                    f"Owner: `{action['owner']}`{approval}  \n"
                    f"{action['rationale']}  \n"
                    f"*Expected impact: {action['expected_impact']}*"
                )
                st.divider()

            if recommendation["do_not_do"]:
                st.warning(
                    "**Do not do:**\n\n"
                    + "\n".join(f"- {item}" for item in recommendation["do_not_do"]),
                    icon="⚠️",
                )

            st.markdown("### Retrieved playbooks")
            for hit in data["knowledge"]:
                with st.expander(f"{hit['section']} · score {hit['score']:.3f}", expanded=False):
                    st.caption(f"Source: {hit['document']}")
                    st.write(hit["text"])

        with right:
            st.markdown("### Quality review")
            q1, q2 = st.columns(2)
            q1.metric("Reviewer", "Passed" if review["passed"] else "Needs revision")
            q2.metric("Review score", f"{review['score']:.0%}")
            st.progress(review["score"])
            if review["issues"]:
                st.error("\n".join(f"• {issue}" for issue in review["issues"]))
            else:
                st.success("Grounding, action coverage, and safety checks passed.")

            st.markdown("### Evidence")
            for evidence in recommendation["evidence"]:
                st.markdown(f"- {evidence}")

            st.markdown("### Human decision")
            decision = st.radio(
                "Decision", ["Approve", "Request changes", "Reject"], horizontal=True
            )
            rating = st.slider("Recommendation quality", 1, 5, 4)
            comment = st.text_area(
                "Reviewer notes",
                placeholder="Explain what worked or what should change...",
            )
            if st.button("Submit decision and feedback", use_container_width=True):
                try:
                    api_post(
                        "/v1/feedback",
                        {
                            "analysis_id": data["analysis_id"],
                            "rating": rating,
                            "comment": f"{decision}: {comment}".strip(),
                        },
                    )
                    st.success("Decision saved for evaluation and future improvement.")
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

            if data.get("ticket_id"):
                st.success(f"Approval ticket #{data['ticket_id']} created.", icon="✅")

        with st.expander("Management report"):
            st.markdown(data["report_markdown"])
        with st.expander("Raw workflow response"):
            st.json(data)
    else:
        st.info(
            f"Select a customer and run the workflow. The current highest-risk account is {default_customer}."
        )


elif page == "Agent Evaluation":
    st.title("Agent Evaluation")
    st.caption("Track deterministic quality gates before prompts or models are promoted.")

    payload = api_get("/v1/operations/evaluation")
    agent_eval = payload["agent_evaluation"]
    metrics = agent_eval.get("metrics", {})
    model = payload.get("model", {})

    st.subheader("Agent regression suite")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Golden cases", int(metrics.get("case_count", 0)))
    e2.metric("Risk-band accuracy", f"{metrics.get('risk_band_accuracy', 0):.0%}")
    e3.metric("Action coverage", f"{metrics.get('mean_keyword_coverage', 0):.0%}")
    e4.metric("Reviewer pass rate", f"{metrics.get('review_pass_rate', 0):.0%}")
    e5.metric("Mean latency", f"{metrics.get('mean_latency_ms', 0):.0f} ms")

    cases = pd.DataFrame(agent_eval.get("cases", []))
    if not cases.empty:
        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("#### Latency by golden case")
            latency = cases.set_index("customer_id")[["latency_ms"]]
            st.bar_chart(latency, horizontal=True)
        with chart2:
            st.markdown("#### Review score by golden case")
            scores = cases.set_index("customer_id")[["review_score"]]
            st.bar_chart(scores, horizontal=True)
        st.dataframe(cases, use_container_width=True, hide_index=True)

    st.subheader("Supervised ML model")
    model_metrics = model.get("metrics", model)
    preferred = [
        ("ROC-AUC", "roc_auc"),
        ("Average precision", "average_precision"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("F1", "f1"),
        ("Threshold", "decision_threshold"),
    ]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, preferred, strict=True):
        value = model_metrics.get(key)
        col.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "—")

    st.subheader("Model comparison leaderboard")
    root = _project_root()
    leaderboard_path = root / "artifacts" / "models" / "tuning" / "grid_search" / "leaderboard.json"
    production_metadata_path = root / "artifacts" / "models" / "churn_model_metadata.json"
    leaderboard_data = _read_json_file(leaderboard_path)
    production_metadata = _read_json_file(production_metadata_path) or {}
    registry_metadata = production_metadata.get("registry") or {}
    promoted_run_id = production_metadata.get("mlflow_run_id") or registry_metadata.get(
        "run_id"
    )

    if not leaderboard_data:
        st.info(
            "No leaderboard artifact found yet. Run tuning first to compare all model families in one table."
        )
    else:
        rows: list[dict[str, Any]] = []
        for rank, item in enumerate(leaderboard_data, start=1):
            score = item.get("score")
            rows.append(
                {
                    "Rank": rank,
                    "Model": item.get("model_type", "n/a"),
                    "Metric": item.get("score_metric", "n/a"),
                    "Score": round(float(score), 6) if isinstance(score, (int, float)) else score,
                    "Promoted": bool(promoted_run_id and item.get("mlflow_run_id") == promoted_run_id),
                    "Run ID": item.get("mlflow_run_id", "n/a"),
                    "Best params": json.dumps(item.get("params", {}), separators=(",", ":")),
                }
            )

        leaderboard_df = pd.DataFrame(rows)
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        promoted_rows = leaderboard_df[leaderboard_df["Promoted"] == True]  # noqa: E712
        if not promoted_rows.empty:
            promoted = promoted_rows.iloc[0]
            st.success(
                f"Promoted production model: {promoted['Model']} ({promoted['Metric']}={promoted['Score']})"
            )
        else:
            st.info("The current deployed champion was registered separately from the latest tuning leaderboard.")

    st.subheader("Model registry deployment")
    registry_payload = api_get("/v1/operations/model-registry")
    if not registry_payload.get("enabled"):
        st.info(registry_payload.get("reason", "Model Registry is not enabled."))
    elif registry_payload.get("error"):
        st.error(registry_payload["error"])
    else:
        aliases = registry_payload.get("aliases", {})
        champion = aliases.get("champion") or aliases.get(registry_payload.get("deployment_alias", "champion"))
        previous = aliases.get("previous_champion") or aliases.get(f"previous_{registry_payload.get('deployment_alias', 'champion')}")
        raw_registry_name = registry_payload.get("registered_model_name", "n/a")
        registry_display_name = raw_registry_name.replace("-", " ").title()
        m1, m2, m3 = st.columns(3)
        m1.metric("Registered model", registry_display_name)
        m2.metric("Champion version", champion.get("version", "—") if champion else "—")
        m3.metric("Previous version", previous.get("version", "—") if previous else "—")
        st.caption(f"Registry ID: `{raw_registry_name}` · Deployment URI: `models:/{raw_registry_name}@{registry_payload.get('deployment_alias', 'champion')}`")

        versions = pd.DataFrame(registry_payload.get("versions", []))
        if not versions.empty:
            st.markdown("#### Registry versions")
            st.dataframe(versions, use_container_width=True, hide_index=True)

        events = pd.DataFrame(registry_payload.get("events", []))
        if not events.empty:
            st.markdown("#### Rollout and rollback history")
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.caption("No registry events have been recorded yet.")

    with st.expander("Model metadata and evaluation payload"):
        st.json(payload)

    st.info(
        "Production promotion gate: run this page's underlying evaluation in GitHub Actions whenever "
        "agent prompts, tools, retrieval logic, or the ML model changes."
    )


elif page == "System Trace":
    st.title("System Trace")
    st.caption("Inspect recent workflow executions, operational actions, memory, and feedback.")

    payload = api_get("/v1/operations/traces", {"limit": 100})
    runtime = payload["runtime"]
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Recent analyses", runtime["prediction_count"])
    t2.metric("Tickets", runtime["ticket_count"])
    t3.metric("Feedback records", runtime["feedback_count"])
    t4.metric("Long-term memories", runtime["memory_count"])
    t5.metric("Session messages", runtime["session_message_count"])

    st.caption(
        f"Agent provider: `{runtime['agent_provider']}` · Model: `{runtime.get('model', 'n/a')}` · Provider available: `{runtime.get('provider_available', runtime.get('ollama_available', False))}` · Fallback used: `{runtime.get('fallback_used', False)}`"
    )

    tab1, tab2, tab3 = st.tabs(["Analysis runs", "Approval tickets", "Human feedback"])
    with tab1:
        predictions = pd.DataFrame(payload["predictions"])
        if predictions.empty:
            st.info("Run a customer analysis to generate the first trace.")
        else:
            predictions["probability"] = predictions["probability"].map(lambda x: f"{x:.1%}")
            st.dataframe(predictions, use_container_width=True, hide_index=True)
    with tab2:
        tickets = pd.DataFrame(payload["tickets"])
        if tickets.empty:
            st.info("No approval tickets have been created.")
        else:
            st.dataframe(tickets, use_container_width=True, hide_index=True)
    with tab3:
        feedback = pd.DataFrame(payload["feedback"])
        if feedback.empty:
            st.info("No human feedback has been submitted.")
        else:
            st.dataframe(feedback, use_container_width=True, hide_index=True)

    if st.button("Refresh trace data"):
        api_get.clear()
        st.rerun()
