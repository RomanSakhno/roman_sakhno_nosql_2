import os
import re
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b[\w.-]+\b", str(text).lower())


def rrf_fuse(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env.")
    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
    corpus = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    tokenized = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)

    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]

    for query in queries:
        tokens = tokenize(query)
        bm25_scores = bm25.get_scores(tokens)
        bm25_ids = np.argsort(-bm25_scores)[:CANDIDATE_K].tolist()

        query_vector = model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()
        vector_result = index.query(
            vector=query_vector, top_k=CANDIDATE_K, include_metadata=True
        )
        vector_ids = []
        vector_meta = {}
        for match in vector_result["matches"]:
            paper_id = match["metadata"].get("arxiv_id")
            # Map arXiv ID back to the local row index.
            found = df.index[df["id"].astype(str) == str(paper_id)]
            if len(found):
                doc_idx = int(found[0])
                vector_ids.append(doc_idx)
                vector_meta[doc_idx] = match["metadata"]

        fused = rrf_fuse([bm25_ids, vector_ids])
        print(f"\n========== QUERY: {query} ==========")
        print("\nBM25 top-5:")
        for rank, idx in enumerate(bm25_ids[:TOP_K], 1):
            print(f"{rank}. {df.iloc[idx]['title']}")

        print("\nVector top-5:")
        for rank, idx in enumerate(vector_ids[:TOP_K], 1):
            print(f"{rank}. {df.iloc[idx]['title']}")

        print("\nHybrid RRF top-5:")
        for rank, (idx, score) in enumerate(fused[:TOP_K], 1):
            print(f"{rank}. {df.iloc[idx]['title']} | RRF={score:.6f}")


if __name__ == "__main__":
    main()
