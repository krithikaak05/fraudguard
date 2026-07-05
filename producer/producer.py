import pandas as pd
import json
from confluent_kafka import Producer

conf = {
    'bootstrap.servers': 'localhost:9092'
}

producer = Producer(conf)

df = pd.read_csv(
    '/Users/krithikaannaswamykannan/fraud-pipeline/data/ieee-fraud-detection/train_transaction.csv',
    nrows=10000
)

print(f"Loaded {len(df)} transactions")
print("Starting to send events to Kafka...")

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")

for index, row in df.iterrows():
    transaction = row.fillna(0).to_dict()
    transaction_id = str(int(transaction['TransactionID']))
    message = json.dumps(transaction)

    producer.produce(
        topic='transactions',
        key=transaction_id,
        value=message,
        callback=delivery_report
    )

    producer.poll(0)

    if index % 1000 == 0:
        print(f"Progress: {index} transactions sent")

producer.flush()
print("Done! 10000 transactions sent to Kafka.")
