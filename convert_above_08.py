import os
import sys
os.environ['HADOOP_HOME'] = "C:\\hadoop"
sys.path.append("C:\\hadoop\\bin")

import pyspark
from pyspark.sql import SparkSession

#create spark session
def main():
    spark = SparkSession.builder \
        .appName("DataConverter") \
        .master("local[*]") \
        .config("spark.driver.memory", "12g") \
        .getOrCreate()

    print(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

    csv_path = "weighted_score_above_08.csv"
    parquet_path = "review_above_08.parquet"

    #read csv data
    try:
        df = spark.read.csv(csv_path,
                            header=True,
                            inferSchema=True,
                            quote='"',
                            escape='"',
                            multiLine=True)
        print(" CSV read！")
    except Exception as e:
        print(f" Err reading file: {e}")
        spark.stop()
        return

    print("Converting...")
    #drop unneeded columns
    df = df.drop('steam_china_location')
    df = df.drop('hidden_in_steam_china')
    df.write.mode('overwrite').parquet(parquet_path)
    print('succeeded')
    spark.stop()

if __name__ == "__main__":
    main()