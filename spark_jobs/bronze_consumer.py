from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("FraudPipeline-Bronze") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/Users/krithikaannaswamykannan/fraud-pipeline/warehouse") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Spark session started. Reading from Kafka...")

schema = StructType([
    StructField("TransactionID", DoubleType()),
    StructField("isFraud", DoubleType()),
    StructField("TransactionDT", DoubleType()),
    StructField("TransactionAmt", DoubleType()),
    StructField("ProductCD", StringType()),
    StructField("card1", DoubleType()),
    StructField("card2", DoubleType()),
    StructField("card4", StringType()),
    StructField("card6", StringType()),
    StructField("P_emaildomain", StringType()),
    StructField("C1", DoubleType()),
    StructField("C2", DoubleType()),
    StructField("D1", DoubleType()),
])

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "earliest") \
    .load()

df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("kafka_timestamp")
).select("data.*", "kafka_timestamp")

# Create Bronze Iceberg table if it doesn't exist
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.fraud.bronze_transactions (
        TransactionID DOUBLE,
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
        kafka_timestamp TIMESTAMP
    )
    USING iceberg
""")

print("Bronze Iceberg table ready. Writing stream...")

query = df_parsed.writeStream \
    .format("iceberg") \
    .outputMode("append") \
    .option("path", "local.fraud.bronze_transactions") \
    .option("checkpointLocation", "/Users/krithikaannaswamykannan/fraud-pipeline/warehouse/checkpoints/bronze") \
    .start()

print("Streaming to Iceberg Bronze table. Press Ctrl+C to stop.")
query.awaitTermination()
