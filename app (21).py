
import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming build_graph and generate_data are defined in previous cells
# from langgraph_engine import build_graph
# from synthetic_data import generate_data

st.set_page_config(page_title="Financial Crime SOC", layout="wide")

st.title("🏦 Real-Time Financial Crime Intelligence Platform")

app = build_graph()

# Load Data
df = generate_data(200)

results = []

# Run system
for _, row in df.iterrows():

    output = app.invoke({
        "transaction": row.to_dict()
    })

    results.append({
        **row.to_dict(),
        "fraud_score": output["fraud_score"],
        "aml_score": output["aml_score"],
        "risk_score": output["risk_score"],
        "decision": output["decision"]
    })

final_df = pd.DataFrame(results)

# ---------------- DASHBOARD ---------------- #

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(final_df))
col2.metric("Blocked", len(final_df[final_df["decision"]=="BLOCK"]))
col3.metric("Under Review", len(final_df[final_df["decision"]=="REVIEW"]))
col4.metric("Approved", len(final_df[final_df["decision"]=="APPROVE"]))

st.subheader("📊 Risk Distribution")

fig = px.histogram(final_df, x="risk_score", nbins=20)
st.plotly_chart(fig)

st.subheader("🚨 Recent High Risk Transactions")

st.dataframe(final_df.sort_values("risk_score", ascending=False).head(20))

st.subheader("📌 Decision Breakdown")

fig2 = px.pie(final_df, names="decision")
st.plotly_chart(fig2)
