import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Run scripts/01_prepare_data.py first.")

    df = pd.read_parquet(INPUT_FILE)
    texts = (df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).tolist()

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    os.makedirs("embeddings", exist_ok=True)
    np.save(OUTPUT_FILE, vectors)

    print(f"Processed texts: {len(texts)}")
    print(f"Embedding shape: {vectors.shape}")
    print(f"First embedding norm: {np.linalg.norm(vectors[0]):.6f}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
