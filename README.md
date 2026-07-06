# 🔒 FraudGuard: Real-Time Fraud Detection Platform

*Streaming Pipeline for Instant Transaction Risk Intelligence*

![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.7-231F20?style=flat&logo=apache-kafka&logoColor=white) ![Spark](https://img.shields.io/badge/PySpark-3.5.3-E25A1C?style=flat&logo=apache-spark&logoColor=white) ![Iceberg](https://img.shields.io/badge/Apache%20Iceberg-1.8.1-00C853?style=flat) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.9331-0077B6?style=flat)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dashboard](#-dashboard)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline Details](#-pipeline-details)
- [Model Performance](#-model-performance)
- [Dashboard Features](#-dashboard-features)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)

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

## 📸 Dashboard

### Live Metrics
![FraudGuard Header](5.png)

### Analytics Overview
![Analytics](4.png)

### Spend Level Breakdown
![Spend Level](7.png)

### Key Findings
![Key Findings](6.png)

### Fraud Hotspots by Email Channel
![Fraud Hotspots](3.png)

### Risk Heatmap and Flagged Transactions
![Heatmap](2.png)

### Real-Time Transaction Scoring
![Transaction Scoring](risk_assessment.png)

---

## 🏗️ Architecture
---

## 🔮 Key Design Decisions

- PySpark 3.5.3 with Scala 2.12 connectors chosen for Iceberg compatibility
- Kafka KRaft mode used instead of Zookeeper for a simpler, production-ready local setup
- Hadoop catalog used for Iceberg to avoid additional REST catalog infrastructure
- DuckDB reads Iceberg Parquet files directly in the dashboard, avoiding a running Spark session
- scale_pos_weight applied in XGBoost to handle the 3.5% fraud class imbalance

---

*Built with Kafka · Spark Structured Streaming · Apache Iceberg · XGBoost · DuckDB · Streamlit*
