import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="FraudGuard", layout="wide", page_icon="🔒")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%); }
section[data-testid="stSidebar"] { display: none; }
.metric-card { background: linear-gradient(135deg, #111827 0%, #1a2035 100%); border: 1px solid rgba(99,179,237,0.2); border-radius: 16px; padding: 20px 24px; text-align: center; position: relative; overflow: hidden; margin-bottom: 8px; }
.metric-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.metric-card.danger::before { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-card.success::before { background: linear-gradient(90deg, #10b981, #06b6d4); }
.metric-card.warning::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #6366f1); }
.metric-number { font-size: 2rem; font-weight: 700; color: #f0f4ff; margin: 8px 0 4px; }
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; }
.metric-delta { font-size: 0.8rem; margin-top: 4px; }
.delta-up { color: #10b981; }
.delta-down { color: #ef4444; }
.section-header { font-size: 1rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(99,179,237,0.15); }
.hero-title { font-size: 2.2rem; font-weight: 700; color: #f0f4ff; margin: 0; line-height: 1.2; }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; margin-right: 6px; }
.insight-card { background: #111827; border: 1px solid rgba(99,179,237,0.12); border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
.insight-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.insight-value { font-size: 1.1rem; font-weight: 600; color: #f0f4ff; }
.insight-sub { font-size: 0.78rem; color: #475569; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

WAREHOUSE = "/Users/krithikaannaswamykannan/fraud-pipeline/warehouse/fraud"
SILVER_PATH = f"{WAREHOUSE}/silver_transactions/data/*.parquet"
GOLD_SUMMARY = f"{WAREHOUSE}/gold_overall_summary/data/*.parquet"
GOLD_CARD = f"{WAREHOUSE}/gold_fraud_by_card_network/data/*.parquet"
GOLD_EMAIL = f"{WAREHOUSE}/gold_fraud_by_email_domain/data/*.parquet"
GOLD_TIER = f"{WAREHOUSE}/gold_fraud_by_amount_tier/data/*.parquet"
MODEL_PATH = "/Users/krithikaannaswamykannan/fraud-pipeline/notebooks/model_xgboost.json"
RESULTS_PATH = "/Users/krithikaannaswamykannan/fraud-pipeline/notebooks/model_results.json"

@st.cache_data(ttl=30)
def load_gold_summary():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_SUMMARY}\")").df()
        con.close()
        return df.iloc[0]
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_card():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_CARD}\")").df()
        con.close()
        return df
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_email():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_EMAIL}\")").df()
        con.close()
        return df
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_tier():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_TIER}\")").df()
        con.close()
        return df
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_silver():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{SILVER_PATH}\")").df()
        con.close()
        return df
    except:
        con.close()
        return None

@st.cache_resource
def load_model():
    m = XGBClassifier()
    try:
        m.load_model(MODEL_PATH)
        return m
    except:
        return None

@st.cache_data
def load_results():
    try:
        with open(RESULTS_PATH) as f:
            return json.load(f)
    except:
        return None

summary = load_gold_summary()
gold_card = load_gold_card()
gold_email = load_gold_email()
gold_tier = load_gold_tier()
silver = load_silver()
model = load_model()
model_results = load_results()

col_title, col_right = st.columns([3, 1])
with col_title:
    st.markdown("<div style=\"padding:24px 0 8px;\"><div style=\"display:flex;align-items:center;gap:12px;margin-bottom:6px;\"><span style=\"font-size:2rem;\">&#128274;</span><span class=\"hero-title\">FraudGuard</span></div><div style=\"font-size:0.95rem;color:#64748b;margin-top:4px;\">Real-time fraud detection across every transaction, instantly.</div></div>", unsafe_allow_html=True)
with col_right:
    st.markdown("<div style=\"text-align:right;padding-top:32px;\"><span class=\"live-dot\"></span><span style=\"font-size:0.85rem;color:#34d399;font-weight:500;\">Live Monitoring</span><div style=\"font-size:0.75rem;color:#475569;margin-top:6px;\">Real-time transaction stream</div></div>", unsafe_allow_html=True)
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr style=\"border-color:rgba(99,179,237,0.15);margin:4px 0 20px;\">", unsafe_allow_html=True)

if summary is not None:
    total = int(summary["total_transactions"])
    fraud_count = int(summary["fraud_count"])
    legit_count = int(summary["legit_count"])
    fraud_rate = float(summary["fraud_rate_pct"])
    avg_amount = float(summary["avg_transaction_amt"])
    fraud_avg = float(summary["avg_fraud_amt"])
else:
    total = fraud_count = legit_count = 0
    fraud_rate = avg_amount = fraud_avg = 0.0

best_auc = model_results["best_auc"] if model_results else 0.9331
best_model_name = model_results["best_model"] if model_results else "XGBoost"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"<div class=\"metric-card\"><div class=\"metric-label\">Transactions Monitored</div><div class=\"metric-number\">{total:,}</div><div class=\"metric-delta delta-up\">Active stream</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class=\"metric-card danger\"><div class=\"metric-label\">Fraud Alerts</div><div class=\"metric-number\" style=\"color:#f87171;\">{fraud_count:,}</div><div class=\"metric-delta delta-down\">Flagged events</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class=\"metric-card success\"><div class=\"metric-label\">Cleared</div><div class=\"metric-number\" style=\"color:#34d399;\">{legit_count:,}</div><div class=\"metric-delta delta-up\">Approved</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class=\"metric-card warning\"><div class=\"metric-label\">Fraud Rate</div><div class=\"metric-number\" style=\"color:#fbbf24;\">{fraud_rate:.2f}%</div><div class=\"metric-delta\" style=\"color:#94a3b8;\">Threshold: 5%</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class=\"metric-card purple\"><div class=\"metric-label\">Detection Accuracy</div><div class=\"metric-number\" style=\"color:#a78bfa;\">{round(best_auc*100,2)}%</div><div class=\"metric-delta\" style=\"color:#94a3b8;\">{best_model_name}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "Fraud Patterns", "Intelligence Report", "Score Transaction"])

with tab1:
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("<div class=\"section-header\">Transaction Amount Distribution</div>", unsafe_allow_html=True)
        if silver is not None:
            fraud_amounts = silver[silver["isFraud"]==1]["TransactionAmt"].clip(upper=500)
            legit_amounts = silver[silver["isFraud"]==0]["TransactionAmt"].clip(upper=500).sample(min(1000, legit_count))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=legit_amounts, name="Approved", nbinsx=50, marker_color="rgba(52,211,153,0.6)"))
            fig.add_trace(go.Histogram(x=fraud_amounts, name="Flagged", nbinsx=50, marker_color="rgba(248,113,113,0.7)"))
            fig.update_layout(barmode="overlay", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=280, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Amount ($)", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(title="Volume", gridcolor="rgba(255,255,255,0.05)", color="#64748b"))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class=\"section-header\">Fraud Rate by Card Network</div>", unsafe_allow_html=True)
        if gold_card is not None:
            card_data = gold_card[gold_card["card4"] != "0"].sort_values("fraud_rate_pct", ascending=True)
            fig2 = go.Figure(go.Bar(x=card_data["fraud_rate_pct"], y=card_data["card4"], orientation="h", marker=dict(color=card_data["fraud_rate_pct"], colorscale=[[0,"#1e3a5f"],[0.5,"#7c3aed"],[1.0,"#ef4444"]]), text=[f"{r:.1f}%" for r in card_data["fraud_rate_pct"]], textposition="outside", textfont=dict(color="#94a3b8", size=11)))
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=280, margin=dict(l=0,r=40,t=10,b=0), xaxis=dict(title="Fraud Rate (%)", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(color="#94a3b8"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class=\"section-header\">Transaction Volume by Amount Tier</div>", unsafe_allow_html=True)
    if gold_tier is not None:
        tier_order = ["$0-25","$25-50","$50-100","$100-200","$200-500","$500-1K","$1K+"]
        gold_tier["legit_count"] = gold_tier["total_transactions"] - gold_tier["fraud_count"]
        tier_melt = gold_tier.melt(id_vars="amount_tier", value_vars=["fraud_count","legit_count"], var_name="type", value_name="count")
        tier_melt["type"] = tier_melt["type"].map({"fraud_count":"FRAUD","legit_count":"LEGIT"})
        fig3 = px.bar(tier_melt, x="amount_tier", y="count", color="type", color_discrete_map={"FRAUD":"#f87171","LEGIT":"#34d399"}, barmode="stack", category_orders={"amount_tier": tier_order})
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=240, legend=dict(bgcolor="rgba(0,0,0,0)", title=None), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Amount Tier", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(title="Transactions", gridcolor="rgba(255,255,255,0.05)", color="#64748b"))
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class=\"section-header\">Flagged Transactions by Email Channel</div>", unsafe_allow_html=True)
        if gold_email is not None:
            email_data = gold_email[gold_email["P_emaildomain"] != "0"].head(12).copy()
            email_data["domain"] = email_data["P_emaildomain"]
            fig4 = px.treemap(email_data, path=["domain"], values="fraud_count", color="fraud_rate_pct", color_continuous_scale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1.0,"#ef4444"]])
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=300, coloraxis_showscale=False)
            fig4.update_traces(textfont=dict(color="#f0f4ff", size=12))
            st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.markdown("<div class=\"section-header\">Fraud Exposure by Card Category</div>", unsafe_allow_html=True)
        if silver is not None:
            card6_fraud = silver.groupby("card6").agg(total=("isFraud","count"), fraud=("isFraud","sum")).reset_index()
            card6_fraud["legit"] = card6_fraud["total"] - card6_fraud["fraud"]
            card6_fraud = card6_fraud[card6_fraud["total"] > 10]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(name="Approved", x=card6_fraud["card6"], y=card6_fraud["legit"], marker_color="rgba(52,211,153,0.7)"))
            fig5.add_trace(go.Bar(name="Flagged", x=card6_fraud["card6"], y=card6_fraud["fraud"], marker_color="rgba(248,113,113,0.85)"))
            fig5.update_layout(barmode="stack", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=300, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(color="#64748b"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#64748b"))
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div class=\"section-header\">Risk Heatmap: Card Network vs Email Channel</div>", unsafe_allow_html=True)
    if silver is not None:
        top_emails = silver[silver["isFraud"]==1]["P_emaildomain"].value_counts().head(8).index.tolist()
        hm_df = silver[silver["P_emaildomain"].isin(top_emails) & (silver["card4"] != "0")]
        hm_pivot = hm_df.groupby(["card4","P_emaildomain"])["isFraud"].mean().unstack(fill_value=0)
        hm_pivot.index.name = "Card Network"
        hm_pivot.columns.name = "Email Channel"
        fig6 = px.imshow(hm_pivot, color_continuous_scale=[[0,"#0f172a"],[0.3,"#1e3a5f"],[0.7,"#7c3aed"],[1.0,"#ef4444"]], aspect="auto", text_auto=".2f")
        fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10), height=260, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
        fig6.update_traces(textfont=dict(size=10, color="#f0f4ff"))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("<div class=\"section-header\">High Priority Alerts</div>", unsafe_allow_html=True)
    if silver is not None:
        flagged = silver[silver["isFraud"]==1][["TransactionID","TransactionAmt","card4","card6","P_emaildomain","risk_label"]].sort_values("TransactionAmt", ascending=False).head(20)
        st.dataframe(flagged.rename(columns={"TransactionID":"Transaction ID","TransactionAmt":"Amount ($)","card4":"Card Network","card6":"Card Type","P_emaildomain":"Email Channel","risk_label":"Status"}), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("<div class=\"section-header\">Gold Layer Intelligence Report</div>", unsafe_allow_html=True)
    st.markdown("<div style=\"color:#64748b;font-size:0.85rem;margin-bottom:20px;\">Pre-aggregated business intelligence sourced from the Gold layer. Updated on each pipeline run.</div>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("<div class=\"section-header\">Fraud by Card Network</div>", unsafe_allow_html=True)
        if gold_card is not None:
            display_card = gold_card[gold_card["card4"] != "0"].copy()
            display_card = display_card.rename(columns={"card4":"Card Network","total_transactions":"Total","fraud_count":"Fraud","avg_transaction_amt":"Avg Amount ($)","avg_fraud_amt":"Avg Fraud Amount ($)","fraud_rate_pct":"Fraud Rate (%)"})
            display_card["Fraud"] = display_card["Fraud"].astype(int)
            st.dataframe(display_card[["Card Network","Total","Fraud","Fraud Rate (%)","Avg Amount ($)","Avg Fraud Amount ($)"]].sort_values("Fraud Rate (%)", ascending=False), use_container_width=True, hide_index=True)

    with col_r2:
        st.markdown("<div class=\"section-header\">Fraud by Amount Tier</div>", unsafe_allow_html=True)
        if gold_tier is not None:
            display_tier = gold_tier[["amount_tier","total_transactions","fraud_count","fraud_rate_pct","avg_transaction_amt"]].copy()
            display_tier = display_tier.rename(columns={"amount_tier":"Amount Tier","total_transactions":"Total","fraud_count":"Fraud","fraud_rate_pct":"Fraud Rate (%)","avg_transaction_amt":"Avg Amount ($)"})
            display_tier["Fraud"] = display_tier["Fraud"].astype(int)
            tier_order = ["$0-25","$25-50","$50-100","$100-200","$200-500","$500-1K","$1K+"]
            display_tier["Amount Tier"] = pd.Categorical(display_tier["Amount Tier"], categories=tier_order, ordered=True)
            st.dataframe(display_tier.sort_values("Amount Tier"), use_container_width=True, hide_index=True)

    st.markdown("<div class=\"section-header\">Top Email Channels by Fraud Volume</div>", unsafe_allow_html=True)
    if gold_email is not None:
        display_email = gold_email[gold_email["P_emaildomain"] != "0"].copy()
        display_email = display_email.rename(columns={"P_emaildomain":"Email Channel","total_transactions":"Total","fraud_count":"Fraud","fraud_rate_pct":"Fraud Rate (%)","avg_transaction_amt":"Avg Amount ($)"})
        display_email["Fraud"] = display_email["Fraud"].astype(int)
        st.dataframe(display_email[["Email Channel","Total","Fraud","Fraud Rate (%)","Avg Amount ($)"]].head(15), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("<div class=\"section-header\">Real-Time Transaction Scoring</div>", unsafe_allow_html=True)
    st.markdown("<div style=\"color:#64748b;font-size:0.85rem;margin-bottom:20px;\">Enter transaction details to receive an instant risk assessment.</div>", unsafe_allow_html=True)
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
        card_type = st.selectbox("Card Network", ["visa","mastercard","discover","american express"])
        card_cat = st.selectbox("Card Category", ["debit","credit"])
    with col_i2:
        email = st.selectbox("Email Domain", ["gmail.com","yahoo.com","hotmail.com","outlook.com","protonmail.com","unknown"])
        c1_val = st.slider("Address Count", 0.0, 10.0, 1.0, 0.5)
        c2_val = st.slider("Card Count", 0.0, 10.0, 1.0, 0.5)
    with col_i3:
        d1_val = st.slider("Days Since Last Transaction", 0.0, 100.0, 10.0, 1.0)
        card1_val = st.slider("Card Bin", 1000.0, 18000.0, 9000.0, 100.0)
        card2_val = st.slider("Card Subcode", 100.0, 600.0, 300.0, 10.0)

    if st.button("Run Risk Assessment", use_container_width=True):
        if model is not None:
            le4 = LabelEncoder().fit(["visa","mastercard","discover","american express","unknown"])
            le6 = LabelEncoder().fit(["debit","credit","unknown"])
            le_em = LabelEncoder().fit(["gmail.com","yahoo.com","hotmail.com","outlook.com","protonmail.com","unknown"])
            input_data = pd.DataFrame([{"TransactionAmt":amount,"ProductCD":2,"card1":card1_val,"card2":card2_val,"card3":150.0,"card4":le4.transform([card_type])[0],"card5":226.0,"card6":le6.transform([card_cat])[0],"addr1":315.0,"addr2":87.0,"dist1":0.0,"P_emaildomain":le_em.transform([email])[0],"R_emaildomain":0,"C1":c1_val,"C2":c2_val,"C3":0.0,"C4":0.0,"C5":0.0,"C6":1.0,"C7":0.0,"C8":0.0,"C9":0.0,"C10":0.0,"C11":1.0,"C12":0.0,"C13":1.0,"C14":1.0,"D1":d1_val,"D2":0.0,"D3":0.0,"D4":0.0,"D5":0.0,"M1":1,"M2":1,"M3":1,"M4":1,"M5":1,"M6":1,"M7":0,"M8":0,"M9":0}])
            score = model.predict_proba(input_data)[0][1]
            is_fraud = score > 0.5
            color = "#ef4444" if is_fraud else "#10b981"
            label = "HIGH RISK — BLOCK TRANSACTION" if is_fraud else "LOW RISK — APPROVE TRANSACTION"
            pct = float(f"{score * 100:.1f}")
            _, col_score, _ = st.columns([1,1.2,1])
            with col_score:
                st.markdown(f"<div style=\"background:#0f172a;border:2px solid {color};border-radius:20px;padding:36px;text-align:center;margin:20px 0;\"><div style=\"font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;\">Risk Score</div><div style=\"font-size:4rem;font-weight:800;color:{color};line-height:1;\">{pct}%</div><div style=\"font-size:0.85rem;font-weight:600;color:{color};margin-top:10px;\">{label}</div><div style=\"margin-top:20px;background:rgba(0,0,0,0.4);border-radius:8px;height:10px;overflow:hidden;\"><div style=\"height:100%;width:{pct}%;background:{color};border-radius:8px;\"></div></div></div>", unsafe_allow_html=True)
        else:
            st.warning("Model not loaded. Run train_models.py first.")

st.markdown("<hr style=\"border-color:rgba(99,179,237,0.08);margin-top:32px;\"><div style=\"text-align:center;padding:12px 0;\"><span style=\"font-size:0.73rem;color:#334155;\">FraudGuard · Real-Time Fraud Detection Platform</span></div>", unsafe_allow_html=True)
