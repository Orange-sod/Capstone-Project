import os
import sys
os.environ['HADOOP_HOME'] = "C:\\hadoop"
sys.path.append("C:\\hadoop\\bin")

import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SteamDataPreprocessing") \
    .config("spark.driver.memory", "16g") \
    .config("spark.kryoserializer.buffer.max", "1g") \
    .master("local[*]") \
    .getOrCreate()

print('spark session created')
print(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

PARQUET_PATH = 'all_reviews_raw.parquet'
df = spark.read.parquet(PARQUET_PATH)
print('finished reading')

# Phase 1:(Basic Cleaning)
# Drop columns:
# steam_china_location: indicates which part in China is the review posted, useless.
# hidden_in_steam_china: indicates if the game is published in China, noise
# comment_count: a new review has no comments, unrelated label
# game: we have appid as the identification for the game, this is redundant
cols_to_drop = ["steam_china_location", "hidden_in_steam_china","comment_count","game"]

# limit language to english since longformer only support english version
# drop very short reviews to get rid of memes or meaningless reviews.
cleaned_df = df.drop(*cols_to_drop) \
    .filter(F.col("language") == "english") \
    .filter(F.length(F.col("review")) > 50) \
    .filter(F.col("review").isNotNull()) \
    .filter(F.col("weighted_vote_score").isNotNull())

print('End of Phase 1')

# Phase 2: Graph Indexing
# HGAT needs dense index such as 0, 1, 2...
# Use StringIndexer to project raw id into integer index

# 2.1 User ID -> user_index
user_indexer = StringIndexer(inputCol="author_steamid", outputCol="user_index", handleInvalid="skip")
user_indexer_model = user_indexer.fit(cleaned_df)
cleaned_df = user_indexer_model.transform(cleaned_df)

# 2.2 App ID -> game_index
game_indexer = StringIndexer(inputCol="appid", outputCol="game_index", handleInvalid="skip")
game_indexer_model = game_indexer.fit(cleaned_df)
cleaned_df = game_indexer_model.transform(cleaned_df)


print('End of Phase 2')

# Phase 3: Meta Feature Engineering
# build vector feature needed by Cross-Attention for Query

# 3.1 "Playtime Ratio" (time at review / time at all)
cleaned_df = cleaned_df.withColumn(
    "playtime_ratio",
    F.when(F.col("author_playtime_forever") > 0,
           F.col("author_playtime_at_review") / F.col("author_playtime_forever"))
    .otherwise(0.0)
)

# 3.2 Log Transformation
# use log1p (log(x+1)) to compress extremely big numbers
cleaned_df = cleaned_df.withColumn("log_playtime_forever", F.log1p("author_playtime_forever"))
cleaned_df = cleaned_df.withColumn("log_num_games", F.log1p("author_num_games_owned"))
cleaned_df = cleaned_df.withColumn("log_num_reviews", F.log1p("author_num_reviews"))

# 3.3 convert Boolean value to Int (0/1)
cleaned_df = cleaned_df.withColumn("is_purchased", F.col("steam_purchase").cast(IntegerType()))
cleaned_df = cleaned_df.withColumn("is_free", F.col("received_for_free").cast(IntegerType()))

# 3.4 Early Access & Recommendation (Context Features)
# 'is_early_access' to represent as a context of content
# 'voted_up' to examine the consistency of review.
cleaned_df = cleaned_df.withColumn("is_early_access", F.col("written_during_early_access").cast(IntegerType()))
cleaned_df = cleaned_df.withColumn("voted_recommend", F.col("voted_up").cast(IntegerType()))

print('End of Phase 3')

# Phase 4: Final selection and extraction
# Select all columns needed by pytorch

final_output = cleaned_df.select(
    # ID info
    F.col("recommendationid").alias("review_id"),
    F.col("user_index").cast("long"),  # Ensure Long type，PyTorch Embedding needed
    F.col("game_index").cast("long"),  # Ensure Long type

    # Text info
    "review",         # for longformer

    # Meta Features (Cross-Attention Query Part)
    "playtime_ratio",
    "log_playtime_forever",
    "log_num_games",
    "log_num_reviews",
    "is_purchased",
    "is_free",
    "is_early_access",
    "voted_recommend",

    #auxiliary data to select negative sample for memes
    "votes_funny",
    "votes_up",

    # target label
    "weighted_vote_score" # Target Y
)

print('End of Phase 4')
#store as parquet
final_output.write.mode("overwrite").parquet("processed_reviews_dataset.parquet")

print("Data processed, Schema：")
final_output.printSchema()
spark.stop()