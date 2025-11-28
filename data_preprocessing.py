import polars as pl
import os

PARQUET_PATH = 'all_reviews_raw.parquet'
OUTPUT_PATH = "processed_reviews_dataset_polars.parquet"

print("Polars: Initializing Lazy execution plan...")

# 1. use scan_parquet to start the Lazy mode
# will not read the file, only start up a plan
lf = pl.scan_parquet(PARQUET_PATH)

# Phase 1: Basic Cleaning
# Drop columns:
# steam_china_location: indicates which part in China is the review posted, useless.
# hidden_in_steam_china: indicates if the game is published in China, noise
# comment_count: a new review has no comments, unrelated label
# game: we have appid as the identification for the game, this is redundant
cols_to_drop = ["steam_china_location", "hidden_in_steam_china", "comment_count", "game"]

# build the table
# limit language to english since longformer only support english version
# drop very short reviews to get rid of memes or meaningless reviews.
cleaned_lf = (
    lf.drop(cols_to_drop)
    .filter(pl.col("language") == "english")
    .filter(pl.col("review").str.len_chars() > 50)
    .filter(pl.col("review").is_not_null())
    .filter(pl.col("weighted_vote_score").is_not_null())
)

print('Phase 1 (Cleaning) logic defined.')

# Phase 2: Graph Indexing
# Categorical will automatically build the mapping dictionary, to_physical will convert it to ints like 0, 1, 2...
# .cast(pl.Int64) to ensure it's Long type，following PyTorch demand

cleaned_lf = cleaned_lf.with_columns([
    pl.col("author_steamid").cast(pl.Categorical).to_physical().cast(pl.Int64).alias("user_index"),
    pl.col("appid").cast(pl.Categorical).to_physical().cast(pl.Int64).alias("game_index")
])

print('Phase 2 (Indexing) logic defined.')

# Phase 3: Meta Feature Engineering
cleaned_lf = cleaned_lf.with_columns([
    # 3.1 Playtime Ratio
    pl.when(pl.col("author_playtime_forever") > 0)
    .then(pl.col("author_playtime_at_review") / pl.col("author_playtime_forever"))
    .otherwise(0.0)
    .alias("playtime_ratio"),

    # 3.2 Log Transformation
    pl.col("author_playtime_forever").log1p().alias("log_playtime_forever"),
    pl.col("author_num_games_owned").log1p().alias("log_num_games"),
    pl.col("author_num_reviews").log1p().alias("log_num_reviews"),

    # 3.3 Boolean -> Int (0/1)
    pl.col("steam_purchase").cast(pl.Int32).alias("is_purchased"),
    pl.col("received_for_free").cast(pl.Int32).alias("is_free"),

    # 3.4 Context Features -> Int(0/1)
    pl.col("written_during_early_access").cast(pl.Int32).alias("is_early_access"),
    pl.col("voted_up").cast(pl.Int32).alias("voted_recommend")
])

print('Phase 3 (Feature Eng) logic defined.')

# Phase 4: Final selection
final_output_lf = cleaned_lf.select([
    # ID info
    pl.col("recommendationid").alias("review_id"),
    pl.col("user_index"),
    pl.col("game_index"),

    # Text info
    pl.col("review"),

    # Meta Features
    pl.col("playtime_ratio"),
    pl.col("log_playtime_forever"),
    pl.col("log_num_games"),
    pl.col("log_num_reviews"),
    pl.col("is_purchased"),
    pl.col("is_free"),
    pl.col("is_early_access"),
    pl.col("voted_recommend"),

    # Auxiliary
    pl.col("votes_funny"),
    pl.col("votes_up"),

    # Target
    pl.col("weighted_vote_score")
])
print('Phase 4 (Selection) logic defined.')

# Action
print("Starting Streaming Execution... (Please wait)")

# sink_parquet will deal with data blocks in stream
# the peak memory will be low to avoid OOM err
try:
    final_output_lf.sink_parquet(OUTPUT_PATH)
    print(f"Success! Processed data saved to: {OUTPUT_PATH}")

    # output checking
    print("\nPreview of the first 5 rows:")
    print(pl.read_parquet(OUTPUT_PATH).head(5))

    # print schema to ensure types
    print("\nSchema:")
    print(pl.read_parquet_schema(OUTPUT_PATH))

except Exception as e:
    print(f"An error occurred: {e}")