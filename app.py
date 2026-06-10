import streamlit as st
import pandas as pd
import random
import time
import os

from google import genai
from typing import TypedDict
from langgraph.graph import StateGraph, END

# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Autonomous Financial Crime SOC",
    layout="wide"
)

st.title("🏦 Real-Time Financial Crime SOC (Agentic + LangGraph + HITL)")
st.markdown("## 🛡️ Autonomous Fraud Command Center")

# =========================================================
# SESSION STATE
# =========================================================

if "stats" not in st.session_state:

    st.session_state.stats = {
        "APPROVE": 0,
        "REVIEW": 0,
        "BLOCK": 0,
        "FREEZE": 0
    }

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "running" not in st.session_state:
    st.session_state.running = False

if "txn_counter" not in st.session_state:
    st.session_state.txn_counter = 0

if "customer_risk" not in st.session_state:
    st.session_state.customer_risk = {}

if "actions" not in st.session_state:
    st.session_state.actions = {}

if "narratives" not in st.session_state:
    st.session_state.narratives = []

if "drift_score" not in st.session_state:
    st.session_state.drift_score = 0.10

if "model_metrics" not in st.session_state:

    st.session_state.model_metrics = {
        "precision": 96,
        "recall": 94
    }

if "retraining_log" not in st.session_state:
    st.session_state.retraining_log = []

# =========================================================
# PLACEHOLDERS
# =========================================================

metric_placeholder = st.empty()
feed_placeholder = st.empty()

# =========================================================
# TRANSACTION GENERATOR
# =========================================================

def generate_transaction():

    txn_id = f"T{st.session_state.txn_counter}"
    st.session_state.txn_counter += 1

    customer_id = f"C{random.randint(100,120)}"

    txn = {

        "txn_id": txn_id,
        "customer_id": customer_id,

        "amount": random.randint(100, 20000),

        "velocity": random.randint(1, 15),

        "device_change": random.choice([0,1]),

        "geo_risk": random.choice([0,1]),

        "behavioral_anomaly": random.choice([0,1]),

        "risk_score": 0.0,

        "reasons": [],

        "decision": "APPROVE"
    }

    return txn

# =========================================================
# LANGGRAPH STATE
# =========================================================

class FraudState(TypedDict):

    txn_id: str
    customer_id: str
    amount: int
    velocity: int
    device_change: int
    geo_risk: int
    behavioral_anomaly: int
    risk_score: float
    reasons: list
    decision: str

# =========================================================
# AGENTS
# =========================================================

def amount_agent(state):

    if state["amount"] > 12000:

        state["risk_score"] += 0.4
        state["reasons"].append("High Amount Spike")

    return state

def velocity_agent(state):

    if state["velocity"] > 8:

        state["risk_score"] += 0.3
        state["reasons"].append("Velocity Breach")

    return state

def device_agent(state):

    if state["device_change"] == 1:

        state["risk_score"] += 0.2
        state["reasons"].append("Device Change Detected")

    return state

def geo_agent(state):

    if state["geo_risk"] == 1:

        state["risk_score"] += 0.2
        state["reasons"].append("High Risk Geography")

    return state

def behavior_agent(state):

    if state["behavioral_anomaly"] == 1:

        state["risk_score"] += 0.2
        state["reasons"].append("Behavioral Anomaly")

    return state

def memory_agent(state):

    customer = state["customer_id"]

    old_risk = st.session_state.customer_risk.get(customer, 0)

    if old_risk > 2:

        state["risk_score"] += 0.2
        state["reasons"].append("Repeat Risk Customer")

    st.session_state.customer_risk[customer] = old_risk + 1

    return state

def decision_agent(state):

    risk = state["risk_score"]

    if risk >= 1.2:

        state["decision"] = "FREEZE"

    elif risk >= 0.8:

        state["decision"] = "BLOCK"

    elif risk >= 0.5:

        state["decision"] = "REVIEW"

    else:

        state["decision"] = "APPROVE"

    return state

# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(FraudState)

workflow.add_node("amount", amount_agent)
workflow.add_node("velocity", velocity_agent)
workflow.add_node("device", device_agent)
workflow.add_node("geo", geo_agent)
workflow.add_node("behavior", behavior_agent)
workflow.add_node("memory", memory_agent)
workflow.add_node("decision", decision_agent)

workflow.set_entry_point("amount")

workflow.add_edge("amount", "velocity")
workflow.add_edge("velocity", "device")
workflow.add_edge("device", "geo")
workflow.add_edge("geo", "behavior")
workflow.add_edge("behavior", "memory")
workflow.add_edge("memory", "decision")
workflow.add_edge("decision", END)

app = workflow.compile()

# =========================================================
# METRICS
# =========================================================

with metric_placeholder.container():

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "TOTAL",
        sum(st.session_state.stats.values())
    )

    c2.metric(
        "BLOCK",
        st.session_state.stats["BLOCK"]
    )

    c3.metric(
        "REVIEW",
        st.session_state.stats["REVIEW"]
    )

    c4.metric(
        "FREEZE",
        st.session_state.stats["FREEZE"]
    )

    c5.metric(
        "DRIFT SCORE",
        round(st.session_state.drift_score, 2)
    )

# =========================================================
# CONTROLS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    if st.button("▶ Start Live Stream"):

        st.session_state.running = True

with col2:

    if st.button("⏹ Stop Stream"):

        st.session_state.running = False

# =========================================================
# LIVE STREAM
# =========================================================

if st.session_state.running:

    txn = generate_transaction()

    result = app.invoke(txn)

    decision = result["decision"]

    st.session_state.stats[decision] += 1

    st.session_state.alerts.insert(0, result)

    st.session_state.alerts = st.session_state.alerts[:20]

    # DRIFT INCREASE

    st.session_state.drift_score += random.uniform(0.01, 0.05)

    # MODEL DEGRADATION

    st.session_state.model_metrics["precision"] -= random.uniform(0.1, 0.5)

    st.session_state.model_metrics["recall"] -= random.uniform(0.1, 0.5)

    # SELF HEALING

    if st.session_state.drift_score > 0.75:

        st.session_state.retraining_log.append(

            f"Retraining Triggered at Transaction {result['txn_id']}"
        )

        st.session_state.drift_score = 0.20

        st.session_state.model_metrics["precision"] = 97

        st.session_state.model_metrics["recall"] = 96

    # UPDATE METRICS

    with metric_placeholder.container():

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "TOTAL",
            sum(st.session_state.stats.values())
        )

        c2.metric(
            "BLOCK",
            st.session_state.stats["BLOCK"]
        )

        c3.metric(
            "REVIEW",
            st.session_state.stats["REVIEW"]
        )

        c4.metric(
            "FREEZE",
            st.session_state.stats["FREEZE"]
        )

        c5.metric(
            "DRIFT SCORE",
            round(st.session_state.drift_score, 2)
        )

    # LIVE FEED

    with feed_placeholder.container():

        st.subheader("🚨 Live AI Alert Feed")

        latest_alerts = st.session_state.alerts[:10]

        for idx, r in enumerate(latest_alerts):

            emoji = {

                "APPROVE": "🟢",
                "REVIEW": "⚠️",
                "BLOCK": "🚨",
                "FREEZE": "🧊"
            }

            st.markdown(
                f"""
### {emoji[r['decision']]} {r['decision']} | {r['txn_id']} | Risk={round(r['risk_score'],2)}

Reasons: {' | '.join(r['reasons'])}

Amount: ₹{r['amount']}

Customer: {r['customer_id']}
                """
            )

            action_key = f"{r['txn_id']}_{idx}"

            if r["decision"] in ["BLOCK", "REVIEW", "FREEZE"]:

                if st.button(
                    f"Investigate {r['txn_id']}",
                    key=action_key
                ):

                    st.session_state.actions[r["txn_id"]] = "Investigated"

    time.sleep(1)

    st.rerun()

# =========================================================
# INVESTIGATOR ACTIONS
# =========================================================

st.divider()

st.subheader("📌 Investigator Actions")

if st.session_state.actions:

    st.write(st.session_state.actions)

else:

    st.info("No investigations initiated yet.")

# =========================================================
# AI INVESTIGATION NARRATIVE
# =========================================================

st.divider()

st.subheader("🧠 AI Investigation Narrative")

if st.session_state.alerts:

    latest_case = st.session_state.alerts[0]

    try:

        response = client.models.generate_content(

            model="gemini-2.0-flash",

            contents=f"""

            Analyze this suspicious financial transaction.

            Transaction ID:
            {latest_case['txn_id']}

            Decision:
            {latest_case['decision']}

            Risk Score:
            {latest_case['risk_score']}

            Reasons:
            {' | '.join(latest_case['reasons'])}

            Generate enterprise fraud investigation narrative.
            """
        )

        st.success(response.text)

    except Exception:

        st.warning(
            "Gemini narrative unavailable."
        )

else:

    st.info("No investigation narratives available.")

# =========================================================
# HIGH RISK CUSTOMERS
# =========================================================

st.divider()

st.subheader("📊 High Risk Customers")

if st.session_state.customer_risk:

    risk_df = pd.DataFrame(

        list(st.session_state.customer_risk.items()),

        columns=["Customer", "Risk Count"]
    )

    risk_df = risk_df.sort_values(

        by="Risk Count",
        ascending=False
    )

    st.dataframe(
        risk_df.head(10),
        use_container_width=True
    )

else:

    st.info("No customer risk data yet.")

# =========================================================
# DECISION ANALYTICS
# =========================================================

st.divider()

st.subheader("📈 Decision Analytics")

chart_df = pd.DataFrame({

    "Decision": [
        "APPROVE",
        "REVIEW",
        "BLOCK",
        "FREEZE"
    ],

    "Count": [

        st.session_state.stats["APPROVE"],
        st.session_state.stats["REVIEW"],
        st.session_state.stats["BLOCK"],
        st.session_state.stats["FREEZE"]
    ]
})

st.bar_chart(
    chart_df.set_index("Decision")
)

# =========================================================
# CONTINUOUS MODEL EVALUATION
# =========================================================

st.divider()

st.subheader("📈 Continuous AI Model Evaluation")

metrics_df = pd.DataFrame({

    "Cycle": ["C1","C2","C3","C4","C5"],

    "Precision": [
        98,
        97,
        96,
        95,
        round(st.session_state.model_metrics["precision"],2)
    ],

    "Recall": [
        97,
        96,
        95,
        94,
        round(st.session_state.model_metrics["recall"],2)
    ]
})

st.line_chart(
    metrics_df.set_index("Cycle")
)

# =========================================================
# SELF HEALING AI
# =========================================================

st.divider()

st.subheader("🤖 Self-Healing AI Agent")

if st.session_state.retraining_log:

    for log in st.session_state.retraining_log[-5:]:

        st.success(log)

else:

    st.info("No retraining events yet.")

# =========================================================
# ARIZE OBSERVABILITY
# =========================================================

st.divider()

st.subheader("📡 Arize AI Observability")

obs_df = pd.DataFrame({

    "Metric": [
        "Data Drift",
        "Prediction Drift",
        "Recall Degradation",
        "Alert Volume"
    ],

    "Score": [

        round(st.session_state.drift_score,2),

        round(random.uniform(0.1,0.8),2),

        round(
            100 - st.session_state.model_metrics["recall"],
            2
        ),

        len(st.session_state.alerts)
    ]
})

st.dataframe(
    obs_df,
    use_container_width=True
)

if st.session_state.drift_score > 0.40:

    st.error(
        "🚨 Arize Observability Alert: High Drift Detected"
    )

# =========================================================
# GOOGLE CLOUD AGENT BUILDER
# =========================================================

st.divider()

st.subheader("☁️ Google Cloud Agent Builder")

st.success(
    """
Enterprise Investigation Agents Active

✅ Fraud Investigation Agent

✅ Drift Monitoring Agent

✅ Governance Approval Agent

✅ Self-Healing Retraining Agent
"""
)
