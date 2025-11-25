import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, length
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder \
    .appName("SteamSniffTest") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print(" SparkSession created！")

parquet_path = "all_reviews_raw.parquet"

df = spark.read.parquet(parquet_path)
df.select(
    "appid",
).describe().show()


# target_df = df.filter((col("weighted_vote_score") > 0.8) & (col("review").isNotNull())) \
#     .withColumn("review_len", length(col("review")))
#
# sample_pdf = target_df.sample(fraction=0.01).select("review_len").toPandas()
# filtered_pdf = sample_pdf[sample_pdf['review_len'] > 50]
# plt.figure(figsize=(10, 6))
# plt.hist(sample_pdf['review_len'], bins=100, range=(0, 5000), color='#607c8e', alpha=0.7, rwidth=0.85)
# plt.title('Distribution of Steam Review Lengths (Score > 0.8)')
# plt.xlabel('Length (Characters)')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.5)
#
# plt.figure(figsize=(10, 6))
# plt.hist(filtered_pdf['review_len'], bins=100, range=(50, 5000),
#          color='#607c8e', alpha=0.7, rwidth=0.85, log=True)
# plt.title('Distribution of Steam Review Lengths (Length > 50, Log Scale)')
# plt.xlabel('Length (Characters)')
# plt.ylabel('Frequency (Log Scale)')
# plt.grid(axis='y', alpha=0.5)
# plt.show()


plt.show()
spark.stop()