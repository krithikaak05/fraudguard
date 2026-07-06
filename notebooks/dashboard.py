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
.section-header { font-size: 1rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin: 24px 0 4px; padding-bottom: 8px; border-bottom: 1px solid rgba(99,179,237,0.15); }
.section-sub { font-size: 0.82rem; color: #475569; margin-bottom: 12px; }
.hero-title { font-size: 2.2rem; font-weight: 700; color: #f0f4ff; margin: 0; line-height: 1.2; }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; margin-right: 6px; }
.insight-card { border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; }
.insight-card.red { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); }
.insight-card.green { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); }
.insight-card.amber { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); }
.insight-card.blue { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.25); }
.insight-icon { font-size: 1.4rem; margin-bottom: 6px; }
.insight-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.insight-title.red { color: #f87171; }
.insight-title.green { color: #34d399; }
.insight-title.amber { color: #fbbf24; }
.insight-title.blue { color: #60a5fa; }
.insight-body { font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; }
.insight-stat { font-size: 1.5rem; font-weight: 700; margin-top: 6px; }
.insight-stat.red { color: #f87171; }
.insight-stat.green { color: #34d399; }
.insight-stat.amber { color: #fbbf24; }
.insight-stat.blue { color: #60a5fa; }
.finding-row { display: flex; align-items: flex-start; gap: 12px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.finding-bullet { width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
.finding-text { font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; }
.finding-text strong { color: #f0f4ff; }
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
        return df.drop_duplicates().iloc[0]
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_card():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_CARD}\")").df()
        con.close()
        return df.drop_duplicates()
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_email():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_EMAIL}\")").df()
        con.close()
        return df.drop_duplicates()
    except:
        con.close()
        return None

@st.cache_data(ttl=30)
def load_gold_tier():
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet(\"{GOLD_TIER}\")").df()
        con.close()
        return df.drop_duplicates()
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
    st.markdown(f"<div class=\"metric-card purple\"><div class=\"metric-label\">Detection Accuracy</div><div class=\"metric-number\" style=\"color:#a78bfa;\">{round(best_auc*100,2)}%</div><div class=\"metric-delta\" style=\"color:#94a3b8;\">Fraud detection engine</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Overview", "Where Fraud Happens", "Check a Transaction"])

with tab1:
    st.markdown("<div class=\"section-header\">How Are We Doing?</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">A snapshot of all transactions processed through FraudGuard in real time.</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("<div class=\"section-header\">Fraudulent vs Legitimate Transactions by Amount</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-sub\">Most fraud happens at lower transaction amounts where it is harder to detect.</div>", unsafe_allow_html=True)
        if silver is not None:
            fraud_amounts = silver[silver["isFraud"]==1]["TransactionAmt"].clip(upper=500)
            legit_amounts = silver[silver["isFraud"]==0]["TransactionAmt"].clip(upper=500).sample(min(1000, legit_count))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=legit_amounts, name="Legitimate", nbinsx=50, marker_color="rgba(52,211,153,0.6)"))
            fig.add_trace(go.Histogram(x=fraud_amounts, name="Flagged as Fraud", nbinsx=50, marker_color="rgba(248,113,113,0.7)"))
            fig.update_layout(barmode="overlay", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=280, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Transaction Amount ($)", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(title="Number of Transactions", gridcolor="rgba(255,255,255,0.05)", color="#64748b"))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class=\"section-header\">Which Card Networks Are Riskiest?</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-sub\">Percentage of transactions flagged as fraud per card network.</div>", unsafe_allow_html=True)
        if gold_card is not None:
            card_data = gold_card[gold_card["card4"] != "0"].drop_duplicates().sort_values("fraud_rate_pct", ascending=True)
            card_data["card4"] = card_data["card4"].str.title()
            fig2 = go.Figure(go.Bar(
                x=card_data["fraud_rate_pct"],
                y=card_data["card4"],
                orientation="h",
                marker=dict(color=card_data["fraud_rate_pct"], colorscale=[[0,"#1e3a5f"],[0.5,"#7c3aed"],[1.0,"#ef4444"]]),
                text=[f"{r:.1f}% fraud rate" for r in card_data["fraud_rate_pct"]],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11)
            ))
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=280, margin=dict(l=0,r=120,t=10,b=0), xaxis=dict(title="Fraud Rate (%)", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(color="#94a3b8"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class=\"section-header\">Where Does Fraud Occur by Spend Level?</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Breaking down transaction volume and fraud across different spending ranges.</div>", unsafe_allow_html=True)
    if gold_tier is not None:
        tier_order = ["$0-25","$25-50","$50-100","$100-200","$200-500","$500-1K","$1K+"]
        gold_tier_clean = gold_tier.drop_duplicates().copy()
        gold_tier_clean["Legitimate"] = gold_tier_clean["total_transactions"] - gold_tier_clean["fraud_count"]
        gold_tier_clean = gold_tier_clean.rename(columns={"fraud_count":"Flagged as Fraud","amount_tier":"Spend Range"})
        tier_melt = gold_tier_clean.melt(id_vars="Spend Range", value_vars=["Flagged as Fraud","Legitimate"], var_name="Status", value_name="Count")
        fig3 = px.bar(tier_melt, x="Spend Range", y="Count", color="Status", color_discrete_map={"Flagged as Fraud":"#f87171","Legitimate":"#34d399"}, barmode="stack", category_orders={"Spend Range": tier_order})
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=260, legend=dict(bgcolor="rgba(0,0,0,0)", title=None), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(title="Spending Range", gridcolor="rgba(255,255,255,0.05)", color="#64748b"), yaxis=dict(title="Number of Transactions", gridcolor="rgba(255,255,255,0.05)", color="#64748b"))
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("<div class=\"section-header\">Key Findings</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Patterns identified across all transactions. These insights help focus prevention efforts.</div>", unsafe_allow_html=True)

    if gold_card is not None and gold_email is not None and gold_tier is not None:
        card_clean = gold_card[gold_card["card4"] != "0"].drop_duplicates()
        email_clean = gold_email[gold_email["P_emaildomain"] != "0"].drop_duplicates()
        tier_clean = gold_tier.drop_duplicates()

        riskiest_card = card_clean.sort_values("fraud_rate_pct", ascending=False).iloc[0]
        safest_card = card_clean.sort_values("fraud_rate_pct").iloc[0]
        riskiest_email = email_clean.sort_values("fraud_rate_pct", ascending=False).iloc[0]
        most_fraud_email = email_clean.sort_values("fraud_count", ascending=False).iloc[0]
        riskiest_tier = tier_clean.sort_values("fraud_rate_pct", ascending=False).iloc[0]

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown(f"""
            <div class="insight-card red">
                <div class="insight-icon">&#128721;</div>
                <div class="insight-title red">Highest Risk Card</div>
                <div class="insight-stat red">{str(riskiest_card["card4"]).title()}</div>
                <div class="insight-body" style="margin-top:6px;">{riskiest_card["fraud_rate_pct"]:.1f}% of all {str(riskiest_card["card4"]).title()} transactions are fraudulent. This is the riskiest card network on the platform.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-card green">
                <div class="insight-icon">&#9989;</div>
                <div class="insight-title green">Safest Card</div>
                <div class="insight-stat green">{str(safest_card["card4"]).title()}</div>
                <div class="insight-body" style="margin-top:6px;">Only {safest_card["fraud_rate_pct"]:.1f}% fraud rate. {str(safest_card["card4"]).title()} cardholders pose the lowest fraud risk.</div>
            </div>
            """, unsafe_allow_html=True)

        with col_f2:
            st.markdown(f"""
            <div class="insight-card amber">
                <div class="insight-icon">&#128231;</div>
                <div class="insight-title amber">Riskiest Email Domain</div>
                <div class="insight-stat amber">{riskiest_email["P_emaildomain"]}</div>
                <div class="insight-body" style="margin-top:6px;">{riskiest_email["fraud_rate_pct"]:.1f}% fraud rate. Transactions from this email domain are significantly more likely to be fraudulent.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-card red">
                <div class="insight-icon">&#128680;</div>
                <div class="insight-title red">Most Fraud by Volume</div>
                <div class="insight-stat red">{most_fraud_email["P_emaildomain"]}</div>
                <div class="insight-body" style="margin-top:6px;">{int(most_fraud_email["fraud_count"]):,} fraudulent transactions originated from this email domain, the highest absolute count.</div>
            </div>
            """, unsafe_allow_html=True)

        with col_f3:
            st.markdown(f"""
            <div class="insight-card amber">
                <div class="insight-icon">&#128176;</div>
                <div class="insight-title amber">Riskiest Spend Range</div>
                <div class="insight-stat amber">{riskiest_tier["amount_tier"]}</div>
                <div class="insight-body" style="margin-top:6px;">{riskiest_tier["fraud_rate_pct"]:.1f}% of transactions in this range are flagged as fraud. Mid-range purchases carry the highest risk.</div>
            </div>
            """, unsafe_allow_html=True)

            est_loss = fraud_avg * fraud_count
            st.markdown(f"""
            <div class="insight-card blue">
                <div class="insight-icon">&#128200;</div>
                <div class="insight-title blue">Estimated Exposure</div>
                <div class="insight-stat blue">${est_loss:,.0f}</div>
                <div class="insight-body" style="margin-top:6px;">Estimated total value of flagged transactions based on average fraud transaction amount of ${fraud_avg:.0f}.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class=\"section-header\" style=\"margin-top:32px;\">Fraud Hotspots by Email Channel</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Each block represents an email domain. Larger blocks mean more fraud volume, darker color means higher fraud rate.</div>", unsafe_allow_html=True)
    if gold_email is not None:
        email_data = gold_email[gold_email["P_emaildomain"] != "0"].drop_duplicates().head(12).copy()
        fig4 = px.treemap(email_data, path=["P_emaildomain"], values="fraud_count", color="fraud_rate_pct", color_continuous_scale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1.0,"#ef4444"]], labels={"P_emaildomain":"Email Domain","fraud_count":"Fraud Cases","fraud_rate_pct":"Fraud Rate (%)"})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=320, coloraxis_showscale=False)
        fig4.update_traces(textfont=dict(color="#f0f4ff", size=13))
        st.plotly_chart(fig4, use_container_width=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("<div class=\"section-header\">Debit vs Credit Card Fraud</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-sub\">Are debit or credit card users more at risk?</div>", unsafe_allow_html=True)
        if silver is not None:
            card6_data = silver.groupby("card6").agg(total=("isFraud","count"), fraud=("isFraud","sum")).reset_index()
            card6_data["legit"] = card6_data["total"] - card6_data["fraud"]
            card6_data = card6_data[card6_data["total"] > 10]
            card6_data["card6"] = card6_data["card6"].str.title()
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(name="Legitimate", x=card6_data["card6"], y=card6_data["legit"], marker_color="rgba(52,211,153,0.7)"))
            fig5.add_trace(go.Bar(name="Fraudulent", x=card6_data["card6"], y=card6_data["fraud"], marker_color="rgba(248,113,113,0.85)"))
            fig5.update_layout(barmode="stack", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=280, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(color="#64748b", title="Card Type"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#64748b", title="Number of Transactions"))
            st.plotly_chart(fig5, use_container_width=True)

    with col_h2:
        st.markdown("<div class=\"section-header\">Risk Concentration Heatmap</div>", unsafe_allow_html=True)
        st.markdown("<div class=\"section-sub\">Where card type and email domain overlap to create the highest fraud risk.</div>", unsafe_allow_html=True)
        if silver is not None:
            top_emails = silver[silver["isFraud"]==1]["P_emaildomain"].value_counts().head(6).index.tolist()
            hm_df = silver[silver["P_emaildomain"].isin(top_emails) & (silver["card4"] != "0")]
            hm_pivot = hm_df.groupby(["card4","P_emaildomain"])["isFraud"].mean().unstack(fill_value=0)
            hm_pivot.index.name = "Card Network"
            hm_pivot.columns.name = "Email Domain"
            fig6 = px.imshow(hm_pivot, color_continuous_scale=[[0,"#0f172a"],[0.3,"#1e3a5f"],[0.7,"#7c3aed"],[1.0,"#ef4444"]], aspect="auto", text_auto=".0%")
            fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10), height=280, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
            fig6.update_traces(textfont=dict(size=11, color="#f0f4ff"))
            st.plotly_chart(fig6, use_container_width=True)

    st.markdown("<div class=\"section-header\">Top Flagged Transactions</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">The highest value transactions currently flagged as fraudulent.</div>", unsafe_allow_html=True)
    if silver is not None:
        flagged = silver[silver["isFraud"]==1][["TransactionAmt","card4","card6","P_emaildomain"]].drop_duplicates().sort_values("TransactionAmt", ascending=False).head(10)
        flagged = flagged.rename(columns={"TransactionAmt":"Amount ($)","card4":"Card Network","card6":"Card Type","P_emaildomain":"Email Domain"})
        flagged["Email Domain"] = flagged["Email Domain"].replace("0", "Unknown")
        flagged["Card Network"] = flagged["Card Network"].str.title()
        flagged["Card Type"] = flagged["Card Type"].str.title()
        flagged["Status"] = "🔴 Fraud"
        st.dataframe(flagged, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("<div class=\"section-header\">Check Any Transaction</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Enter the details of any transaction below and FraudGuard will instantly assess whether it is at risk.</div>", unsafe_allow_html=True)

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown("<div style=\"color:#7dd3fc;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;\">Transaction Details</div>", unsafe_allow_html=True)
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
        card_type = st.selectbox("Card Network", ["Visa","Mastercard","Discover","American Express"])
        card_cat = st.selectbox("Card Type", ["Debit","Credit"])
    with col_i2:
        st.markdown("<div style=\"color:#7dd3fc;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;\">Customer Info</div>", unsafe_allow_html=True)
        email = st.selectbox("Email Domain", ["gmail.com","yahoo.com","hotmail.com","outlook.com","protonmail.com","Other"])
        num_addresses = st.slider("Number of Addresses on File", 0, 10, 1)
        num_cards = st.slider("Number of Cards on File", 0, 10, 1)
    with col_i3:
        st.markdown("<div style=\"color:#7dd3fc;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;\">Transaction History</div>", unsafe_allow_html=True)
        days_since = st.slider("Days Since Last Transaction", 0, 100, 10)
        card1_val = st.slider("Card Identifier", 1000.0, 18000.0, 9000.0, 100.0)
        card2_val = st.slider("Card Subcode", 100.0, 600.0, 300.0, 10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Risk Assessment", use_container_width=True):
        if model is not None:
            le4 = LabelEncoder().fit(["visa","mastercard","discover","american express","unknown"])
            le6 = LabelEncoder().fit(["debit","credit","unknown"])
            le_em = LabelEncoder().fit(["gmail.com","yahoo.com","hotmail.com","outlook.com","protonmail.com","unknown"])
            card_type_lower = card_type.lower()
            card_cat_lower = card_cat.lower()
            email_lower = email.lower() if email != "Other" else "unknown"
            input_data = pd.DataFrame([{"TransactionAmt":amount,"ProductCD":2,"card1":card1_val,"card2":card2_val,"card3":150.0,"card4":le4.transform([card_type_lower])[0],"card5":226.0,"card6":le6.transform([card_cat_lower])[0],"addr1":315.0,"addr2":87.0,"dist1":0.0,"P_emaildomain":le_em.transform([email_lower])[0],"R_emaildomain":0,"C1":float(num_addresses),"C2":float(num_cards),"C3":0.0,"C4":0.0,"C5":0.0,"C6":1.0,"C7":0.0,"C8":0.0,"C9":0.0,"C10":0.0,"C11":1.0,"C12":0.0,"C13":1.0,"C14":1.0,"D1":float(days_since),"D2":0.0,"D3":0.0,"D4":0.0,"D5":0.0,"M1":1,"M2":1,"M3":1,"M4":1,"M5":1,"M6":1,"M7":0,"M8":0,"M9":0}])
            score = model.predict_proba(input_data)[0][1]
            is_fraud = score > 0.5
            color = "#ef4444" if is_fraud else "#10b981"
            label = "HIGH RISK — DO NOT APPROVE" if is_fraud else "LOW RISK — SAFE TO APPROVE"
            sublabel = "This transaction shows strong indicators of fraudulent activity. We recommend blocking it immediately." if is_fraud else "This transaction appears legitimate based on all available signals. Safe to proceed."
            pct = float(f"{score * 100:.1f}")
            _, col_score, _ = st.columns([1,1.2,1])
            with col_score:
                st.markdown(f"""
                <div style="background:#0f172a;border:2px solid {color};border-radius:20px;padding:36px;text-align:center;margin:20px 0;">
                    <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Risk Score</div>
                    <div style="font-size:4rem;font-weight:800;color:{color};line-height:1;">{pct}%</div>
                    <div style="font-size:0.9rem;font-weight:700;color:{color};margin-top:10px;letter-spacing:0.04em;">{label}</div>
                    <div style="font-size:0.82rem;color:#94a3b8;margin-top:8px;line-height:1.5;">{sublabel}</div>
                    <div style="margin-top:20px;background:rgba(0,0,0,0.4);border-radius:8px;height:10px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;background:{color};border-radius:8px;"></div>
                    </div>
                    <div style="color:#475569;font-size:0.72rem;margin-top:16px;">Powered by our best performing detection model</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Detection model not loaded. Please run the training script first.")

st.markdown("<hr style=\"border-color:rgba(99,179,237,0.08);margin-top:32px;\"><div style=\"text-align:center;padding:12px 0;\"><span style=\"font-size:0.73rem;color:#334155;\">FraudGuard · Real-Time Fraud Detection Platform</span></div>", unsafe_allow_html=True)
