import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env before running this script.")

    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)
    if len(df) != len(embeddings):
        raise ValueError(f"Dataset rows ({len(df)}) != embeddings ({len(embeddings)}).")
    if embeddings.shape[1] != VECTOR_DIM:
        raise ValueError(f"Expected {VECTOR_DIM}-dimensional vectors, got {embeddings.shape[1]}.")

    pc = Pinecone(api_key=api_key)
    existing = {item["name"] for item in pc.list_indexes()}
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )

    index = pc.Index(INDEX_NAME)
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upserting vectors"):
        end = min(start + BATCH_SIZE, len(df))
        batch = []
        for row_num in range(start, end):
            row = df.iloc[row_num]
            batch.append({
                "id": f"paper_{row_num}",
                "values": embeddings[row_num].tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"])[:1000],
                    "abstract": str(row["abstract"])[:500],
                    "authors": str(row["authors"])[:200],
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                },
            })
        index.upsert(vectors=batch)

    print(f"Index: {INDEX_NAME}")
    print(f"Total vectors: {index.describe_index_stats().get('total_vector_count', 'unknown')}")


if __name__ == "__main__":
    main()
