import time
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


CSV_FILE_PATH = "all_reviews_raw.parquet"
TARGET_COLUMN = "weighted_vote_score"
FEATURE_COLUMN = "hidden_in_steam_china"

print('Stop Point 1')

spark = SparkSession.builder \
    .appName("FeatureAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print('Stop Point 2')


raw_df = spark.read.parquet(CSV_FILE_PATH, header=True)

df = raw_df.select(
    F.col(FEATURE_COLUMN).cast(IntegerType()).alias(FEATURE_COLUMN),
    F.col(TARGET_COLUMN).cast(DoubleType()).alias(TARGET_COLUMN)
)

df = df.dropna()
df.cache()
total_rows = df.count()

print('Stop Point 3')
correlation_value = df.stat.corr(FEATURE_COLUMN, TARGET_COLUMN)
print(f"'{FEATURE_COLUMN}' and '{TARGET_COLUMN}' correlation value: {correlation_value:.4f}")

print('Stop Point 4')

# try:
#     stats_df = df.groupBy(FEATURE_COLUMN) \
#         .agg(
#         F.mean(TARGET_COLUMN).alias("mean"),
#         F.stddev(TARGET_COLUMN).alias("stddev"),
#         F.count(TARGET_COLUMN).alias("count")
#     ) \
#         .collect()
#
#     stats_0 = next((row for row in stats_df if row[FEATURE_COLUMN] == 0), None)
#     stats_1 = next((row for row in stats_df if row[FEATURE_COLUMN] == 1), None)
#
#     if not stats_0 or not stats_1 or stats_0['count'] < 2 or stats_1['count'] < 2:
#         print("insufficient data")
#     else:
#         t_stat, p_value = stats.ttest_ind_from_stats(
#             mean1=stats_0['mean'],
#             std1=stats_0['stddev'],
#             nobs1=stats_0['count'],
#             mean2=stats_1['mean'],
#             std2=stats_1['stddev'],
#             nobs2=stats_1['count'],
#             equal_var=False  # Welch's T-test
#         )
#
#         print(f"P-value is: {p_value}")
#
# except Exception as e:
#     print(f"error: {e}")
#

spark.stop()