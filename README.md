readme = """# 🔒 FraudGuard: Real-Time Fraud Detection Platform

*Streaming Pipeline for Instant Transaction Risk Intelligence*

![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.7-231F20?style=flat&logo=apache-kafka&logoColor=white)
![Spark](https://img.shields.io/badge/PySpark-3.5.3-E25A1C?style=flat&logo=apache-spark&logoColor=white)
![Iceberg](https://img.shields.io/badge/Apache%20Iceberg-1.8.1-00C853?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.9331-0077B6?style=flat)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Pipeline Details](#-pipeline-details)
- [Model Performance](#-model-performance)
- [Dashboard Features](#-dashboard-features)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)

---

## 📘 Overview

**FraudGuard** is an end-to-end real-time streaming pipeline that ingests financial transactions through Apache Kafka, processes them through a **Bronze-Silver-Gold medallion architecture** using Spark Structured Streaming and Apache Iceberg, scores them for fraud risk, and surfaces actionable intelligence through a live boardroom-grade monitoring dashboard.

**Key Highlights:**

- 🔄 Real-time event streaming with Apache Kafka (3 partitions, KRaft mode)
- 🏗️ Full Bronze-Silver-Gold medallion architecture on Apache Iceberg
- 🤖 Multi-model comparison with automatic best model selection (AUC: 0.9331)
- 📊 Executive-grade Streamlit dashboard backed by DuckDB
- ⚡ Sub-second event processing latency
- 🎯 82% recall on fraudulent transactions

---

## 🏗️ Architecture

```
Transaction Events
       │
       ▼
 Kafka (3 partitions, KRaft)
       │
       ▼
Spark Structured Streaming
       │
       ▼
Apache Iceberg Bronze Layer  ──►  Raw events, Parquet on disk
       │
       ▼
Apache Iceberg Silver Layer  ──►  Deduplicated, cleaned, MD5 hashed
       │
       ▼
Apache Iceberg Gold Layer    ──►  Pre-aggregated business intelligence
       │
       ▼
XGBoost Fraud Scorer         ──►  AUC: 0.9331, Recall: 82%
       │
       ▼
Streamlit Dashboard (DuckDB) ──►  Live monitoring and risk scoring
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Event Streaming | Apache Kafka 3.7 (KRaft, Docker) |
| Stream Processing | PySpark 3.5.3 Structured Streaming |
| Storage Format | Apache Iceberg 1.8.1 + Parquet |
| Query Layer | DuckDB |
| Fraud Scoring | XGBoost, Random Forest, Logistic Regression |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker Desktop |

---

## ⚙️ Pipeline Details

### Bronze Layer
- Kafka consumer reads transaction events in micro-batches via Spark Structured Streaming
- Writes raw events to Iceberg table with Kafka timestamp preserved
- 3 Parquet files written, one per Kafka partition
- 42,338 raw events captured

### Silver Layer
- Deduplicates on TransactionID, reducing 42,338 rows to 32,338 unique records
- Fills nulls with sensible defaults (email domain, card type)
- Adds `risk_label` column (FRAUD / LEGIT)
- Adds MD5 `row_hash` for data lineage tracking

### Gold Layer
- **Fraud by Card Network:** fraud rate and average transaction amount per network
- **Fraud by Email Domain:** top 20 domains ranked by fraud volume and rate
- **Fraud by Amount Tier:** fraud concentration across 7 spending ranges
- **Overall Summary:** total transactions, fraud count, fraud rate, average amounts

---

## 🤖 Model Performance

Three models were trained on 590K transactions and compared. The best performing model was automatically selected and deployed to the dashboard.

| Model | AUC Score |
|---|---|
| **XGBoost** | **0.9331** ✅ Selected |
| Random Forest | 0.9232 |
| Logistic Regression | 0.7540 |

- **Training set:** 472,432 transactions
- **Test set:** 118,108 transactions
- **Fraud recall:** 82% (catches 4 out of 5 fraudulent transactions)
- **Class imbalance handling:** `scale_pos_weight` applied (fraud rate: 3.5%)

---

## 📊 Dashboard Features

**Overview Tab**
- Five live metric cards: transactions monitored, fraud alerts, cleared, fraud rate, detection accuracy
- Transaction amount distribution: fraud vs legitimate overlay histogram
- Fraud rate by card network: gradient bar chart
- Transaction volume by spend tier: stacked bar breakdown

**Where Fraud Happens Tab**
- Six key finding cards: highest risk card, safest card, riskiest email domain, most fraud by volume, riskiest spend range, estimated financial exposure
- Fraud hotspot treemap by email channel
- Debit vs credit card fraud exposure
- Risk concentration heatmap: card network vs email domain
- Top 10 highest value flagged transactions

**Check a Transaction Tab**
- Enter any transaction details and receive an instant fraud probability score
- Plain English verdict: approve or block recommendation
- Visual risk score bar with percentage

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Total Transactions Processed | 32,338 |
| Fraud Alerts Generated | 921 |
| Overall Fraud Rate | 2.85% |
| Detection Accuracy | 93.31% |
| Estimated Fraud Exposure | $130,239 |
| Highest Risk Card Network | Discover (3.9% fraud rate) |
| Highest Risk Email Domain | frontiernet.net (35.7% fraud rate) |
| Highest Risk Spend Range | $500-1K (4.03% fraud rate) |

---

## 📂 Project Structure

```
fraudguard/
├── docker-compose.yml              # Kafka KRaft setup
├── producer/
│   └── producer.py                 # Kafka transaction producer
├── spark_jobs/
│   ├── bronze_consumer.py          # Kafka → Iceberg Bronze streaming
│   ├── silver_transformer.py       # Bronze → Silver batch transform
│   └── gold_aggregator.py          # Silver → Gold aggregation
├── notebooks/
│   ├── train_models.py             # Multi-model training and comparison
│   ├── model_xgboost.json          # Trained XGBoost model
│   ├── model_results.json          # Model comparison results
│   └── dashboard.py                # Streamlit dashboard
└── .gitignore
```
