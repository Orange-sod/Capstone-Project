import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from transformers import LongformerTokenizer


class SteamReviewDataset(Dataset):
    def __init__(self, parquet_path, tokenizer_name="allenai/longformer-base-4096", max_length=2048):
        """
        Args:
            parquet_path: path to processed parquet file
            tokenizer_name: HuggingFace model name
            max_length: max length for text (2048 or 4096)
        """
        print(f"Loading data from {parquet_path}...")

        # 1. read files using polars
        df = pl.read_parquet(parquet_path)

        # 2. convert columns to Numpy lists (to implement O(1) random read speed)
        # keep content data as list for processing
        self.reviews = df["review"].to_list()

        # ID indexing (for HGAT)
        self.user_indices = df["user_index"].to_numpy().astype(np.int64)
        self.game_indices = df["game_index"].to_numpy().astype(np.int64)

        # Label (target)
        self.labels = df["weighted_vote_score"].to_numpy().astype(np.float32)

        # Meta Features (for Cross-Attention Query)
        # combine all features into a matrix [N_samples, N_features]
        meta_cols = [
            "playtime_ratio", "log_playtime_forever", "log_num_games",
            "log_num_reviews", "is_purchased", "is_free",
            "is_early_access", "voted_recommend"
        ]
        # select().to_numpy() directly return 2D matrix
        self.meta_features = df.select(meta_cols).to_numpy().astype(np.float32)

        # auxiliary data (for calculating Loss weight or negative sampling)
        self.votes_funny = df["votes_funny"].to_numpy().astype(np.float32)
        self.votes_up = df["votes_up"].to_numpy().astype(np.float32)

        # 3. intialize Tokenizer
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = LongformerTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        print(f"Dataset loaded. Total samples: {len(self.reviews)}")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # 1.  Tokenization
        text = self.reviews[idx]

        # Longformer's Tokenizer will automatically deal with [CLS], [SEP] and padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",  # padding to max length for batch processing
            truncation=True,  # truncate over-long content
            return_tensors="pt"  # return PyTorch Tensor
        )

        # tokenizer returns [1, seq_len]，we need [seq_len]
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 2. get other features
        user_idx = torch.tensor(self.user_indices[idx], dtype=torch.long)
        game_idx = torch.tensor(self.game_indices[idx], dtype=torch.long)
        meta_feats = torch.tensor(self.meta_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        # 3. calculate funny_ratio (for Loss weighting)
        # avoid divide by 0
        total_votes = self.votes_up[idx] + self.votes_funny[idx]
        funny_ratio = 0.0
        if total_votes > 0:
            funny_ratio = self.votes_funny[idx] / total_votes
        funny_ratio = torch.tensor(funny_ratio, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "user_index": user_idx,
            "game_index": game_idx,
            "meta_features": meta_feats,
            "label": label,
            "funny_ratio": funny_ratio
        }


# --- test code ---
if __name__ == "__main__":

    DATA_PATH = "processed_reviews_polars.parquet"

    # initialize Dataset
    # max_length set as 512 just for speed
    dataset = SteamReviewDataset(DATA_PATH, max_length=512)

    # get a sample for cheking
    sample = dataset[0]

    print("\n--- Sample Verification ---")
    print(f"Input IDs Shape: {sample['input_ids'].shape}")  # should be [512]
    print(f"User Index: {sample['user_index']}")
    print(f"Meta Features Shape: {sample['meta_features'].shape}")  # should be [8]
    print(f"Label: {sample['label']}")
    print(f"Funny Ratio: {sample['funny_ratio']}")

    print("\nDataset test passed!")