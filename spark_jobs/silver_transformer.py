from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, md5, concat_ws
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("FraudPipeline-Silver") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/Users/krithikaannaswamykannan/fraud-pipeline/warehouse") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Reading from Bronze Iceberg table...")

df_bronze = spark.read.format("iceberg").load("local.fraud.bronze_transactions")

print(f"Bronze row count: {df_bronze.count()}")

# Step 1: Drop duplicates on TransactionID
df_deduped = df_bronze.dropDuplicates(["TransactionID"])

# Step 2: Cast TransactionID to integer
df_clean = df_deduped.withColumn("TransactionID", col("TransactionID").cast("integer"))

# Step 3: Fill nulls with sensible defaults
df_clean = df_clean \
    .fillna({"P_emaildomain": "unknown", "card4": "unknown", "card6": "unknown"}) \
    .fillna(0.0)

# Step 4: Add a fraud risk label column
df_clean = df_clean.withColumn(
    "risk_label",
    when(col("isFraud") == 1.0, lit("FRAUD")).otherwise(lit("LEGIT"))
)

# Step 5: Add a row hash for deduplication tracking
df_clean = df_clean.withColumn(
    "row_hash",
    md5(concat_ws("_",
        col("TransactionID").cast("string"),
        col("TransactionAmt").cast("string"),
        col("TransactionDT").cast("string")
    ))
)

print(f"Silver row count after dedup: {df_clean.count()}")
print(f"Fraud transactions: {df_clean.filter(col('isFraud') == 1.0).count()}")
print(f"Legit transactions: {df_clean.filter(col('isFraud') == 0.0).count()}")

# Create Silver Iceberg table
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.silver_transactions (
        TransactionID INT,
        isFraud DOUBLE,
        TransactionDT DOUBLE,
        TransactionAmt DOUBLE,
        ProductCD STRING,
        card1 DOUBLE,
        card2 DOUBLE,
        card4 STRING,
        card6 STRING,
        P_emaildomain STRING,
        C1 DOUBLE,
        C2 DOUBLE,
        D1 DOUBLE,
        kafka_timestamp TIMESTAMP,
        risk_label STRING,
        row_hash STRING
    )
    USING iceberg
""")

# Write to Silver
df_clean.select(
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card4", "card6",
    "P_emaildomain", "C1", "C2", "D1", "kafka_timestamp",
    "risk_label", "row_hash"
).writeTo("local.fraud.silver_transactions").append()

print("Silver layer written successfully.")
spark.stop()
