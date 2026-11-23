
import json
import os

import pandas as pd
from tqdm import tqdm

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000


def extract_year(paper: dict) -> int:
    try:
        versions = paper.get("versions", [])
        if versions:
            return int(versions[0]["created"].split()[3])
    except (IndexError, ValueError, KeyError, TypeError):
        pass

    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    parsed = paper.get("authors_parsed", [])
    if parsed:
        names = []
        for entry in parsed[:10]:
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                names.append(f"{last} {first}".strip())
        return ", ".join(names)

    return paper.get("authors", "").replace("\\n", " ")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    os.makedirs("data", exist_ok=True)

    records = []
    total_valid = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Reading arXiv metadata"):
            line = line.strip()
            if not line:
                continue

            paper = json.loads(line)

            title = paper.get("title", "").replace("\\n", " ").strip()
            abstract = paper.get("abstract", "").replace("\\n", " ").strip()

            if not title or not abstract:
                continue

            total_valid += 1

            record = {
                "id": paper["id"],
                "title": title,
                "abstract": abstract,
                "authors": format_authors(paper),
                "year": extract_year(paper),
                "category": (
                    paper.get("categories", "unknown").split() or ["unknown"]
                )[0],
            }

            # Reservoir sampling: retain a representative sample
            # without loading the entire dataset into memory.
            if len(records) < MAX_RECORDS:
                records.append(record)
            else:
                import random
                index = random.randrange(total_valid)
                if index < MAX_RECORDS:
                    records[index] = record

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid records were extracted.")

    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nTotal valid papers scanned: {total_valid}")
    print(f"Sampled papers: {len(df)}")

    print("\nTop 10 categories:")
    print(df["category"].value_counts().head(10))

    print("\nYear distribution:")
    print(df["year"].value_counts().sort_index())

    print("\nFirst record:")
    print(df.iloc[0].to_dict())

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
