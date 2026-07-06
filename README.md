# 🔒 FraudGuard

**Real-time fraud detection across every transaction, instantly.**

FraudGuard is an end-to-end streaming data pipeline that ingests financial transactions in real time, processes them through a Bronze-Silver-Gold medallion architecture, scores them for fraud risk, and surfaces insights through a live monitoring dashboard.

---

## Architecture

```
Transaction Events
       │
       ▼
 Kafka (3 partitions)
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

## Tech Stack

| Layer | Technology |
|---|---|
| Event Streaming | Apache Kafka (KRaft, Docker) |
| Stream Processing | PySpark 3.5.3 Structured Streaming |
| Storage Format | Apache Iceberg + Parquet |
| Query Layer | DuckDB |
| Fraud Scoring | XGBoost, Random Forest, Logistic Regression |
| Dashboard | Streamlit + Plotly |

---

## Model Performance

Three models were trained and compared. The best performing model was automatically selected.

| Model | AUC Score |
|---|---|
| XGBoost | **0.9331** |
| Random Forest | 0.9232 |
| Logistic Regression | 0.7540 |

XGBoost achieved 82% recall on fraudulent transactions with a 3.5% fraud rate in the dataset.

---

## Pipeline Details

**Bronze Layer**
- Kafka consumer reads transaction events in micro-batches
- Writes raw events to Iceberg table with Kafka timestamp
- 3 Parquet files, one per Kafka partition

**Silver Layer**
- Deduplicates on TransactionID
- Fills nulls with sensible defaults
- Adds risk label column (FRAUD / LEGIT)
- Adds MD5 row hash for lineage tracking
- Bronze 42,338 rows reduced to Silver 32,338 unique rows

**Gold Layer**
- Fraud rate by card network
- Fraud rate by email domain
- Fraud rate by transaction amount tier
- Overall pipeline summary metrics

---

## Dashboard Features

- Five live metric cards: transactions monitored, fraud alerts, cleared, fraud rate, detection accuracy
- Overview tab: transaction amount distribution, fraud rate by card network, spend tier breakdown
- Where Fraud Happens tab: key findings cards, fraud hotspot treemap, debit vs credit breakdown, risk concentration heatmap, top flagged transactions
- Check a Transaction tab: enter any transaction details and get an instant fraud probability score with plain English recommendation

---

## Project Structure

```
fraudguard/
├── docker-compose.yml
├── producer/
│   └── producer.py
├── spark_jobs/
│   ├── bronze_consumer.py
│   ├── silver_transformer.py
│   └── gold_aggregator.py
├── notebooks/
│   ├── train_models.py
│   ├── model_xgboost.json
│   ├── model_results.json
│   └── dashboard.py
└── .gitignore
```

---

## How to Run

**1. Start Kafka**
```bash
docker compose up -d
```

**2. Send transactions**
```bash
python producer/producer.py
```

**3. Start Bronze streaming job**
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.8.1 \
  spark_jobs/bronze_consumer.py
```

**4. Run Silver transform**
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.8.1 \
  spark_jobs/silver_transformer.py
```

**5. Run Gold aggregation**
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.8.1 \
  spark_jobs/gold_aggregator.py
```

**6. Train models**
```bash
python notebooks/train_models.py
```

**7. Launch dashboard**
```bash
streamlit run notebooks/dashboard.py
```

---

## Key Design Decisions

- PySpark 3.5.3 with Scala 2.12 connectors chosen for Iceberg compatibility
- Kafka KRaft mode used instead of Zookeeper for a simpler local setup
- Hadoop catalog used for Iceberg to avoid additional infrastructure
- DuckDB reads Iceberg Parquet files directly, avoiding a running Spark session in the dashboard
- scale_pos_weight applied in XGBoost training to handle the 3.5% fraud class imbalance

---

*Built with Kafka, Spark Structured Streaming, Apache Iceberg, XGBoost, DuckDB, and Streamlit.*
