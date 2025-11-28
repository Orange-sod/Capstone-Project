import polars as pl
import os
import time

def main():
    # path configuration
    csv_path = "all_reviews.csv"
    parquet_path = "all_reviews_raw.parquet"

    print(f"Polars: Starting conversion from {csv_path} to {parquet_path}...")
    start_time = time.time()

    try:
        lf = pl.scan_csv(
            csv_path,
            infer_schema_length=10000,
            ignore_errors=False,
            quote_char='"',
        )

        print("Schema inferred. Starting streaming conversion...")
        lf.sink_parquet(parquet_path)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Success! Conversion completed in {duration:.2f} seconds.")
        print(f"Output saved to: {parquet_path}")

        file_size = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"File Size: {file_size:.2f} MB")

    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
