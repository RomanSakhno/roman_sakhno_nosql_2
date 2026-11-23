import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
MAX_WORDS = 180
OVERLAP = 30
BATCH_SIZE = 200


def fixed_chunks(text, size=MAX_WORDS, overlap=OVERLAP):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def sentence_chunks(text, size=MAX_WORDS):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current and count + len(words) > size:
            chunks.append(" ".join(current))
            current, count = [], 0
        # A single sentence longer than size is split to respect model input limits.
        if len(words) > size:
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            for i in range(0, len(words), size):
                chunks.append(" ".join(words[i:i + size]))
            continue
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def ensure_index(pc, name):
    names = {item["name"] for item in pc.list_indexes()}
    if name not in names:
        pc.create_index(
            name=name, dimension=VECTOR_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
    return pc.Index(name)


def upload_chunks(index, records, model, label):
    texts = [r["chunk_text"] for r in records]
    vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Upsert {label}"):
        batch = []
        for i in range(start, min(start + BATCH_SIZE, len(records))):
            r = records[i]
            batch.append({
                "id": r["id"],
                "values": vectors[i].tolist(),
                "metadata": {k: v for k, v in r.items() if k != "id"},
            })
        index.upsert(vectors=batch)


def search_chunks(index, model, query, label):
    q = model.encode([query], normalize_embeddings=True)[0].tolist()
    result = index.query(vector=q, top_k=5, include_metadata=True)
    print(f"\n--- {label}: {query} ---")
    for rank, match in enumerate(result["matches"], 1):
        meta = match["metadata"]
        print(
            f"{rank}. {meta.get('title')} | chunk={meta.get('chunk_number')} | "
            f"score={match['score']:.4f} | {meta.get('chunk_text', '')[:250]}"
        )


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in .env.")
    df = pd.read_parquet("data/arxiv_subset.parquet")
    selected = df.assign(word_count=df["abstract"].str.split().str.len()).nlargest(30, "word_count")
    pc = Pinecone(api_key=api_key)
    model = SentenceTransformer(MODEL_NAME)
    fixed_index = ensure_index(pc, FIXED_INDEX)
    semantic_index = ensure_index(pc, SEMANTIC_INDEX)

    fixed_records, semantic_records = [], []
    for row_idx, row in selected.reset_index(drop=True).iterrows():
        base = {
            "arxiv_id": str(row["id"]), "title": str(row["title"])[:1000],
            "year": int(row["year"]), "category": str(row["category"]),
        }
        for chunk_num, chunk in enumerate(fixed_chunks(str(row["abstract"]))):
            fixed_records.append({
                **base, "id": f"fixed_{row_idx}_{chunk_num}",
                "chunk_text": chunk[:3500], "chunk_number": int(chunk_num),
            })
        for chunk_num, chunk in enumerate(sentence_chunks(str(row["abstract"]))):
            semantic_records.append({
                **base, "id": f"semantic_{row_idx}_{chunk_num}",
                "chunk_text": chunk[:3500], "chunk_number": int(chunk_num),
            })

    print(f"Fixed-size chunks: {len(fixed_records)}")
    print(f"Sentence-aware chunks: {len(semantic_records)}")
    upload_chunks(fixed_index, fixed_records, model, "fixed")
    upload_chunks(semantic_index, semantic_records, model, "sentence-aware")

    for query in [
        "deep learning for image recognition",
        "reinforcement learning and decision making",
        "large language models and natural language processing",
    ]:
        search_chunks(fixed_index, model, query, "Fixed-size")
        search_chunks(semantic_index, model, query, "Sentence-aware")


if __name__ == "__main__":
    main()
