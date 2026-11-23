import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5


def encode_query(model, query):
    return model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype(np.float32)


def show_matches(title, matches, df):
    print(f"\n--- {title} ---")
    for rank, match in enumerate(matches, 1):
        meta = match.get("metadata", {})
        arxiv_id = meta.get("arxiv_id")
        print(
            f"{rank}. {meta.get('title', '(no title)')} | "
            f"{meta.get('category')} | {meta.get('year')} | "
            f"score={match.get('score', 0):.4f} | "
            f"{str(meta.get('abstract', ''))[:220]}"
        )


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env.")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_parquet("data/arxiv_subset.parquet")

    query = "teaching machines to recognize objects in pictures"
    vector = encode_query(model, query)
    result = index.query(vector=vector.tolist(), top_k=TOP_K, include_metadata=True)
    show_matches("Pure semantic search", result["matches"], df)

    from datetime import datetime
    recent_year = datetime.now().year - 5
    recent = index.query(
        vector=vector.tolist(), top_k=TOP_K, include_metadata=True,
        filter={"category": {"$eq": "cs.LG"}, "year": {"$gte": recent_year}},
    )
    show_matches(f"Filtered: cs.LG, year >= {recent_year}", recent["matches"], df)

    older = index.query(
        vector=vector.tolist(), top_k=TOP_K, include_metadata=True,
        filter={"year": {"$lt": 2015}},
    )
    show_matches("Filtered: before 2015", older["matches"], df)

    print("\n--- Local metric comparison ---")
    vectors = np.load("embeddings/embeddings.npy").astype(np.float32)
    q = vector / np.linalg.norm(vector)
    cosine = vectors @ q
    dot = vectors @ q
    l2 = np.linalg.norm(vectors - q[None, :], axis=1)
    metrics = {
        "cosine (higher is better)": np.argsort(-cosine)[:TOP_K],
        "dot product (higher is better)": np.argsort(-dot)[:TOP_K],
        "L2 distance (lower is better)": np.argsort(l2)[:TOP_K],
    }
    for metric_name, indices in metrics.items():
        print(f"\n{metric_name}")
        for rank, idx in enumerate(indices, 1):
            row = df.iloc[int(idx)]
            score = cosine[idx] if metric_name.startswith(("cosine", "dot")) else l2[idx]
            print(f"{rank}. {row['title']} | {row['category']} | {row['year']} | {score:.6f}")


if __name__ == "__main__":
    main()
