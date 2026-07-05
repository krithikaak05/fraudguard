from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, round, when, lit

spark = SparkSession.builder \
    .appName("FraudPipeline-Gold") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/Users/krithikaannaswamykannan/fraud-pipeline/warehouse") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Reading Silver layer...")
df = spark.read.format("iceberg").load("local.fraud.silver_transactions")
print(f"Silver rows: {df.count()}")

# Gold Table 1: Fraud summary by card network
print("Building gold_fraud_by_card_network...")
gold_card = df.groupBy("card4").agg(
    count("*").alias("total_transactions"),
    sum("isFraud").alias("fraud_count"),
    round(avg("TransactionAmt"), 2).alias("avg_transaction_amt"),
    round(avg(when(col("isFraud")==1, col("TransactionAmt"))), 2).alias("avg_fraud_amt"),
    round((sum("isFraud") / count("*")) * 100, 2).alias("fraud_rate_pct")
).orderBy(col("fraud_rate_pct").desc())

spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.gold_fraud_by_card_network (
        card4 STRING,
        total_transactions LONG,
        fraud_count DOUBLE,
        avg_transaction_amt DOUBLE,
        avg_fraud_amt DOUBLE,
        fraud_rate_pct DOUBLE
    ) USING iceberg
""")
gold_card.writeTo("local.fraud.gold_fraud_by_card_network").overwritePartitions()
print("gold_fraud_by_card_network done.")
gold_card.show()

# Gold Table 2: Fraud summary by email domain
print("Building gold_fraud_by_email_domain...")
gold_email = df.groupBy("P_emaildomain").agg(
    count("*").alias("total_transactions"),
    sum("isFraud").alias("fraud_count"),
    round((sum("isFraud") / count("*")) * 100, 2).alias("fraud_rate_pct"),
    round(avg("TransactionAmt"), 2).alias("avg_transaction_amt")
).orderBy(col("fraud_count").desc()).limit(20)

spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.gold_fraud_by_email_domain (
        P_emaildomain STRING,
        total_transactions LONG,
        fraud_count DOUBLE,
        fraud_rate_pct DOUBLE,
        avg_transaction_amt DOUBLE
    ) USING iceberg
""")
gold_email.writeTo("local.fraud.gold_fraud_by_email_domain").overwritePartitions()
print("gold_fraud_by_email_domain done.")
gold_email.show()

# Gold Table 3: Fraud summary by amount tier
print("Building gold_fraud_by_amount_tier...")
df_tiered = df.withColumn("amount_tier",
    when(col("TransactionAmt") <= 25, "$0-25")
    .when(col("TransactionAmt") <= 50, "$25-50")
    .when(col("TransactionAmt") <= 100, "$50-100")
    .when(col("TransactionAmt") <= 200, "$100-200")
    .when(col("TransactionAmt") <= 500, "$200-500")
    .when(col("TransactionAmt") <= 1000, "$500-1K")
    .otherwise("$1K+")
)
gold_tier = df_tiered.groupBy("amount_tier").agg(
    count("*").alias("total_transactions"),
    sum("isFraud").alias("fraud_count"),
    round((sum("isFraud") / count("*")) * 100, 2).alias("fraud_rate_pct"),
    round(avg("TransactionAmt"), 2).alias("avg_transaction_amt")
).orderBy(col("fraud_rate_pct").desc())

spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.gold_fraud_by_amount_tier (
        amount_tier STRING,
        total_transactions LONG,
        fraud_count DOUBLE,
        fraud_rate_pct DOUBLE,
        avg_transaction_amt DOUBLE
    ) USING iceberg
""")
gold_tier.writeTo("local.fraud.gold_fraud_by_amount_tier").overwritePartitions()
print("gold_fraud_by_amount_tier done.")
gold_tier.show()

# Gold Table 4: Overall summary metrics
print("Building gold_overall_summary...")
total = df.count()
fraud_total = int(df.filter(col("isFraud")==1).count())
legit_total = total - fraud_total
avg_amt = df.agg(round(avg("TransactionAmt"), 2)).collect()[0][0]
fraud_avg_amt = df.filter(col("isFraud")==1).agg(round(avg("TransactionAmt"), 2)).collect()[0][0]

import builtins
summary = spark.createDataFrame([{
    "total_transactions": total,
    "fraud_count": fraud_total,
    "legit_count": legit_total,
    "fraud_rate_pct": builtins.round(fraud_total / total * 100, 2),
    "avg_transaction_amt": float(avg_amt),
    "avg_fraud_amt": float(fraud_avg_amt)
}])

spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.gold_overall_summary (
        total_transactions LONG,
        fraud_count LONG,
        legit_count LONG,
        fraud_rate_pct DOUBLE,
        avg_transaction_amt DOUBLE,
        avg_fraud_amt DOUBLE
    ) USING iceberg
""")
summary.writeTo("local.fraud.gold_overall_summary").overwritePartitions()
print("gold_overall_summary done.")
summary.show()

print("\nAll Gold tables built successfully.")
spark.stop()
