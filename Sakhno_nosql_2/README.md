# NoSQL — Homework 2: Semantic Search with Pinecone

**Author:** Sakhno  
**Dataset:** arXiv Metadata — Cornell University (Kaggle)  
**Embedding Model:** `allenai/specter2_base`  
**Embedding Framework:** Sentence Transformers  
**Vector Database:** Pinecone  
**Vector Dimension:** 768  
**Number of Papers:** 10,000

---

# 1. Project Overview

This project implements a semantic search system for scientific publications using the arXiv dataset, transformer-based embeddings, and Pinecone vector database.

The main objective is to explore vector search, metadata filtering, similarity metrics, document chunking strategies, and hybrid retrieval combining semantic and keyword search.

The implementation covers all required assignment parts (1–4 and 6), as well as the optional hybrid-search implementation (part 5).

## Implemented Features

- Downloading and preprocessing arXiv metadata.
- Sampling 10,000 scientific papers from the complete dataset.
- Converting JSONL data into Parquet format.
- Generating normalized 768-dimensional embeddings.
- Creating and populating a Pinecone vector index.
- Performing semantic similarity searches.
- Filtering results using metadata.
- Comparing Cosine Similarity, Dot Product, and Euclidean Distance.
- Implementing fixed-size and sentence-aware chunking.
- Comparing chunk-based retrieval strategies.
- Combining BM25 and vector search using Reciprocal Rank Fusion (RRF).

---

# 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Dataset | Cornell-University/arxiv |
| Data Processing | Pandas, NumPy |
| Data Storage | Parquet |
| Embedding Model | allenai/specter2_base |
| Embedding Framework | Sentence Transformers |
| Deep Learning | PyTorch |
| Vector Database | Pinecone |
| Keyword Search | BM25 |
| Hybrid Search | Reciprocal Rank Fusion |
| Progress Monitoring | tqdm |

---

# 3. Project Structure

```text
Sakhno_nosql_2/
│
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
│
├── data/
│   └── arxiv_subset.parquet
│
├── embeddings/
│   └── embeddings.npy
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Generated data and embeddings are excluded from Git.

---

# 4. Environment Setup

## Requirements

- Python 3.11+
- Kaggle account and API credentials
- Pinecone account and API key
- Internet connection

## 4.1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4.2. Configure Kaggle access

The dataset was downloaded using Kaggle CLI 2.2.5.

The Kaggle API token can be configured in the current PowerShell session:

```powershell
$env:KAGGLE_API_TOKEN="your-kaggle-token"
```

Verify access:

```powershell
.\.venv\Scripts\kaggle.exe datasets list -s arxiv
```

Download the dataset:

```powershell
.\.venv\Scripts\kaggle.exe datasets download -d Cornell-University/arxiv
```

Extract the archive:

```powershell
Expand-Archive -Path arxiv.zip -DestinationPath . -Force
```

Expected input file:

```text
arxiv-metadata-oai-snapshot.json
```

## 4.3. Configure Pinecone

Create a `.env` file in the project root:

```dotenv
PINECONE_API_KEY=your-real-api-key
```

Never commit `.env`, Kaggle credentials, or other secrets.

---

# 5. Execution Order

All scripts should be executed from the project root.

```powershell
.\.venv\Scripts\python.exe scripts/01_prepare_data.py

.\.venv\Scripts\python.exe scripts/02_embed.py

.\.venv\Scripts\python.exe scripts/03_load_to_pinecone.py

.\.venv\Scripts\python.exe scripts/04_search.py

.\.venv\Scripts\python.exe scripts/05_chunking.py

.\.venv\Scripts\python.exe scripts/06_hybrid_search.py
```

Parts 2–5 require a valid Pinecone API key.

---

# PART 1 — Data Preparation and Embedding

## 1.1. Dataset Preparation

The original dataset was downloaded from:

`Cornell-University/arxiv`

The downloaded archive contained approximately 1.77 GB of compressed data.

The original JSONL file contains millions of scientific publications.

### Data Processing

The preparation script performs the following operations:

1. Reads the original JSONL file line by line.
2. Extracts valid papers with non-empty titles and abstracts.
3. Extracts the original publication year from the first version's creation date.
4. Formats author information.
5. Extracts the first arXiv category.
6. Selects a representative sample of 10,000 papers.
7. Saves the resulting dataset in Parquet format.

Reservoir Sampling is used to select a fixed-size random sample without loading the entire dataset into memory.

This approach is particularly useful when processing large JSONL files.

### Actual Execution Results

| Metric | Result |
|---|---:|
| Total valid papers scanned | 3,261,651 |
| Selected papers | 10,000 |
| Earliest publication year | 1991 |
| Latest publication year | 2026 |
| Output format | Parquet |
| Output file | `data/arxiv_subset.parquet` |

### Publication Year Distribution

The resulting sample covers the complete historical range of the dataset.

Selected year counts:

| Year | Papers |
|---|---:|
| 1991 | 1 |
| 2000 | 97 |
| 2007 | 189 |
| 2010 | 222 |
| 2015 | 304 |
| 2020 | 546 |
| 2023 | 651 |
| 2024 | 733 |
| 2025 | 888 |
| 2026 | 1,044 |

The distribution is not uniform. It reflects the relative representation of publication periods in the source dataset.

### Top Categories

The most represented categories in the resulting sample included:

| Category | Papers |
|---|---:|
| cs.CV | 491 |
| cs.LG | 478 |
| hep-ph | 457 |
| quant-ph | 399 |
| hep-th | 352 |
| astro-ph | 301 |
| cs.CL | 258 |
| cond-mat.mtrl-sci | 237 |
| gr-qc | 219 |
| cond-mat.mes-hall | 210 |

**Output:**

```text
Total valid papers scanned: 3261651
Sampled papers: 10000

Saved: data/arxiv_subset.parquet
```

## 1.2. Embedding Generation

The selected papers are converted into dense vector representations.

The input text is constructed by combining the title and abstract:

```text
title [SEP] abstract
```

The embedding model:

`allenai/specter2_base`

### Model Configuration

The model was loaded through Sentence Transformers:

```python
model = SentenceTransformer("allenai/specter2_base")
```

Sentence Transformers automatically created a model wrapper with mean pooling because the repository did not contain a native SentenceTransformer configuration.

Therefore, this implementation uses the base SPECTER2 model with mean pooling. It does not explicitly load the SPECTER2 proximity adapter.

### Embedding Configuration

| Parameter | Value |
|---|---|
| Model | allenai/specter2_base |
| Batch size | 64 |
| Number of papers | 10,000 |
| Embedding dimension | 768 |
| Normalization | L2 |
| Output format | NumPy |
| Output file | `embeddings/embeddings.npy` |

### Actual Execution Results

```text
Batches: 157/157 [12:35]

Processed texts: 10000
Embedding shape: (10000, 768)
First embedding norm: 1.000000

Saved: embeddings/embeddings.npy
```

### Performance

| Metric | Result |
|---|---:|
| Total batches | 157 |
| Full batches | 156 |
| Last batch | 16 records |
| Total processing time | 12 min 35 sec |
| Average batch time | 4.81 sec |

All generated embeddings were successfully normalized.

## 1.3. Theoretical Questions

### 1. Pinecone vs Qdrant vs Chroma

**Pinecone**

Pinecone is a managed vector database service. The provider manages infrastructure, scaling, availability, and indexing.

Advantages:
- Managed infrastructure.
- Simple API integration.
- No need to operate database servers.
- Suitable for hosted production applications.

Disadvantages:
- Dependence on a commercial hosted platform.
- Pricing and usage limits.
- Less infrastructure control compared with self-hosting.

**Qdrant**

Qdrant is an open-source vector database available for self-hosting and managed cloud deployment.

Advantages:
- Open-source ecosystem.
- Infrastructure and deployment control.
- Flexible filtering capabilities.
- Suitable for production applications.

Disadvantages:
- Self-hosted deployments require operational management.
- Scaling and infrastructure maintenance become the team's responsibility.

**Chroma**

Chroma is an open-source vector database designed for developer productivity, local experimentation, and RAG applications.

Advantages:
- Simple developer experience.
- Convenient local deployment.
- Fast prototyping.
- Easy integration with AI application frameworks.

Disadvantages:
- Different operational and scaling characteristics compared with specialized managed production services.
- Production suitability depends on deployment architecture and workload.

**Conclusion**

| Database | Best Use Case |
|---|---|
| Pinecone | Managed cloud vector search |
| Qdrant | Self-hosting and infrastructure control |
| Chroma | Local development and prototyping |

There is no universally fastest database. Performance depends on dataset size, vector dimensions, indexing configuration, filtering, hardware, network latency, and workload.

A fair comparison requires benchmarking the same dataset and query workload.

### 2. Why SPECTER2 instead of all-MiniLM-L6-v2?

SPECTER2 is designed for scientific-document representation and scientific literature retrieval.

It is trained in the domain of scientific publications and is intended to capture relationships between scholarly documents.

`all-MiniLM-L6-v2` is a compact, general-purpose sentence embedding model.

It is useful for general semantic similarity tasks but is not specifically trained for scientific-paper retrieval.

For arXiv titles and abstracts, SPECTER2 is therefore a more domain-aligned choice.

MiniLM can still be preferable when lower latency, smaller model size, or general-purpose text similarity is more important.

### 3. Recommended Similarity Metric

The SPECTER2 documentation recommends cosine similarity for comparing document embeddings.

Cosine similarity measures the angle between two vectors and focuses on their directional similarity.

In this implementation, embeddings are L2-normalized and Pinecone uses cosine similarity.

---

# PART 2 — Pinecone Index and Vector Upload

## 2.1. Index Configuration

The script creates the following Pinecone index:

| Parameter | Value |
|---|---|
| Index name | `arxiv-papers` |
| Vector dimension | 768 |
| Similarity metric | Cosine |
| Upsert batch size | 200 |
| Total vectors | 10,000 |

Each vector receives a stable ID and metadata.

### Metadata Fields

- `arxiv_id`
- `title`
- `abstract`
- `authors`
- `year`
- `category`

## 2.2. Actual Execution Results

```text
Upserting vectors: 100% | 50/50

Index: arxiv-papers
Total vectors: 10000
```

### Upload Performance

| Metric | Result |
|---|---:|
| Total vectors | 10,000 |
| Batch size | 200 |
| Number of batches | 50 |
| Upload duration | 31 seconds |

All 10,000 vectors were successfully uploaded.

## 2.3. Why Is the Abstract Truncated to 500 Characters?

Pinecone limits the amount of metadata stored per vector. The assignment specifies a 40 KB metadata limit.

Storing complete abstracts in every vector's metadata increases index storage and may exceed metadata limits.

The implementation therefore stores only the first 500 characters of each abstract.

The complete abstract remains available in the local Parquet dataset and can be retrieved using the paper ID.

This separates compact searchable metadata from the complete source record.

---

# PART 3 — Semantic Search and Similarity Metrics

## 3.1. Pure Semantic Search

The semantic search implementation:

1. Accepts a natural-language query.
2. Generates a query embedding using the same model.
3. Sends the vector to Pinecone.
4. Retrieves the nearest vectors.
5. Returns the matching publication metadata.

The system retrieves the Top-5 most similar publications.

Semantic search does not require exact keyword matches. It compares the semantic representations of the query and stored documents.

## 3.2. Metadata Filtering

The implementation demonstrates two metadata-filtered searches.

**Filter 1:**

```text
category = cs.LG
year >= 2021
```

**Filter 2:**

```text
year < 2015
```

Both searches returned five results matching their respective metadata constraints.

This demonstrates that vector similarity search can be combined with structured metadata filtering.

## 3.3. Similarity Metric Comparison

The local experiment compares:

- Cosine Similarity
- Dot Product
- Euclidean Distance (L2)

### Actual Results

All three metrics returned the same Top-5 ranking.

| Rank | Cosine | Dot Product | L2 |
|---:|---:|---:|---:|
| 1 | Rabbit, toad, and the Moon | Same | Same |
| 2 | Artificial neural networks for 3D cell shape recognition | Same | Same |
| 3 | Hilditchs Algorithm Based Tamil Character Recognition | Same | Same |
| 4 | A Face-like Structure Detection on Planet and Satellite Surfaces | Same | Same |
| 5 | Knowledge-Augmented Vision Language Models for Underwater Bioacoustic Spectrogram Analysis | Same | Same |

Example scores:

| Metric | First Result |
|---|---:|
| Cosine Similarity | 0.869681 |
| Dot Product | 0.869681 |
| L2 Distance | 0.510528 |

## 3.4. Why Are Cosine Similarity and Dot Product Equivalent?

Cosine similarity is defined as:

\[
\operatorname{cos}(x,y)=
\frac{x\cdot y}{\|x\|\|y\|}
\]

For normalized vectors:

\[
\|x\|=\|y\|=1
\]

Therefore:

\[
\operatorname{cos}(x,y)=x\cdot y
\]

Cosine similarity and dot product produce identical scores and rankings for unit vectors.

## 3.5. Why Does L2 Produce the Same Ranking?

Euclidean distance is:

\[
\|x-y\|_2^2
=
\|x\|_2^2+\|y\|_2^2-2x\cdot y
\]

For normalized vectors:

\[
\|x-y\|_2^2=2-2\operatorname{cos}(x,y)
\]

Since 2 is constant, minimizing Euclidean distance is equivalent to maximizing cosine similarity.

Consequently, L2, cosine, and dot product produce the same nearest-neighbor ordering for normalized vectors, ignoring numerical precision and ties.

The difference is in score interpretation:

- Cosine and dot product: higher is better.
- L2 distance: lower is better.

## 3.6. What Happens Without Normalization?

Without normalization:

- Dot product depends on both vector direction and magnitude.
- Cosine similarity compares direction independently of magnitude.
- Euclidean distance depends on absolute vector positions and magnitudes.

Therefore, the three metrics can produce different rankings when vectors are not normalized.

---

# PART 4 — Document Chunking

## 4.1. Objective

The chunking experiment investigates how different document segmentation strategies affect semantic retrieval.

The implementation selects the 30 papers with the longest abstracts.

Two chunking strategies are compared:

1. Fixed-size chunking.
2. Sentence-aware chunking.

Each strategy uses a separate Pinecone index.

| Index | Strategy |
|---|---|
| `arxiv-chunks-fixed` | Fixed-size |
| `arxiv-chunks-semantic` | Sentence-aware |

## 4.2. Fixed-Size Chunking

Fixed-size chunking divides text into windows of a predefined number of words.

An overlap between adjacent chunks preserves some context at chunk boundaries.

Advantages:
- Simple implementation.
- Predictable chunk size.
- Easy to control storage and processing requirements.

Disadvantages:
- May split sentences.
- Can separate definitions from explanations.
- May produce incomplete semantic representations.

## 4.3. Sentence-Aware Chunking

Sentence-aware chunking groups complete sentences until the target word limit is reached.

It attempts to preserve grammatical and contextual boundaries.

Advantages:
- Better sentence integrity.
- More coherent local context.
- Reduced risk of splitting a statement.

Disadvantages:
- Variable chunk lengths.
- Sentence splitting can be imperfect.
- Long sentences may exceed the target size.

This implementation uses sentence-boundary-based grouping. It is a semantic approximation, not embedding-based topic segmentation.

## 4.4. Actual Execution Results

```text
Fixed-size chunks: 90
Sentence-aware chunks: 62
```

Both chunk collections were successfully embedded and uploaded to their respective Pinecone indexes.

### Chunk Count Comparison

| Metric | Fixed-Size | Sentence-Aware |
|---|---:|---:|
| Number of chunks | 90 | 62 |
| Embedding generation | Successful | Successful |
| Pinecone upload | Successful | Successful |
| Search | Successful | Successful |

Sentence-aware chunking produced approximately 31% fewer chunks than fixed-size chunking.

## 4.5. Search Experiments

Three queries were used:

1. `deep learning for image recognition`
2. `reinforcement learning and decision making`
3. `large language models and natural language processing`

Both indexes returned Top-5 results for every query.

### Observations

**Query 1: Deep Learning for Image Recognition**

Fixed-size results included:
- Ranking Research Institutions Based On Related Academic Conferences
- Dying Radio Sources in Clusters
- 12CO emission from EP Aqr
- Tracing the Thoughts of a Coding Agent Playing ARC-AGI-3

Sentence-aware results included:
- Large Scale Enrichment and Statistical Cyber Characterization of Network Traffic
- Tracing the Thoughts of a Coding Agent Playing ARC-AGI-3
- Ranking Research Institutions Based On Related Academic Conferences

**Query 2: Reinforcement Learning and Decision Making**

Both approaches retrieved:

`Tracing the Thoughts of a Coding Agent Playing ARC-AGI-3: Lessons for Continual Learning`

This publication appeared in both retrieval strategies.

**Query 3: Large Language Models and NLP**

Both approaches retrieved the coding-agent publication.

Other results differed between strategies, demonstrating that chunk boundaries affect the semantic representation and retrieval results.

## 4.6. Chunking Analysis

### Which Strategy Produces More Meaningful Chunks?

Sentence-aware chunking generally produces more coherent chunks because it attempts to preserve complete sentences.

Fixed-size chunking is predictable and simple, but may split a sentence or separate related information.

However, sentence-aware chunking is not always superior. Sentence boundaries do not necessarily correspond to semantic topic boundaries.

### Are Sentences Split?

Fixed-size chunking can split sentences at arbitrary word boundaries.

This can remove grammatical context and make an embedding less representative of the original statement.

Sentence-aware chunking reduces this problem, although punctuation-based sentence detection is not perfect.

### How Does Overlap Affect Chunk Count and Coverage?

Overlap repeats part of one chunk in the next.

Benefits:
- Preserves context across chunk boundaries.
- Reduces the risk of losing concepts split between chunks.

Trade-offs:
- More chunks.
- Higher embedding and storage costs.
- More duplicate retrieval results.

The optimal overlap depends on document structure and retrieval requirements.

---

# PART 5 — Hybrid Search (Optional)

## 5.1. Objective

Hybrid search combines lexical retrieval and semantic vector retrieval.

The implementation uses:

- BM25 for keyword-based search.
- Pinecone for dense vector search.
- Reciprocal Rank Fusion (RRF) to combine rankings.

BM25 is effective for exact terms, names, and technical expressions.

Vector search is useful for semantic similarity, synonyms, and paraphrased queries.

Hybrid search attempts to combine both strengths.

## 5.2. Reciprocal Rank Fusion

RRF combines ranked result lists without requiring their raw scores to use the same scale.

The formula is:

\[
RRF(d)=\sum_{r\in R}
\frac{1}{k+\operatorname{rank}_r(d)}
\]

Where:

- \(d\) is a document.
- \(R\) is the set of ranked result lists.
- \(k\) is a smoothing constant.
- \(\operatorname{rank}_r(d)\) is the document's position in a result list.

The implementation retrieves candidates from both BM25 and vector search and combines their rankings.

## 5.3. Actual Search Experiments

Three queries were tested.

### Query 1: BERT Fine-Tuning

**BM25 Top Results**

1. Towards Interpreting BERT for Reading Comprehension Based QA
2. Only Train Once: A One-Shot Neural Network Training And Pruning Framework
3. Typoglycemia under the Hood: Investigating Language Models' Understanding of Scrambled Words
4. BTRec: BERT-Based Trajectory Recommendation for Personalized Tours
5. Multimodal Detection of Fake Reviews using BERT and ResNet-50

**Vector Search Top Results**

1. An initial attempt of combining visual selective attention with deep reinforcement learning
2. PHS: A Toolbox for Parallel Hyperparameter Search
3. PRO-Bid: Pareto-Prioritized Regret Optimization for Constraint-Aware Generative Auto-Bidding
4. MCTS-based Automated Negotiation Agent
5. Choice and Market Design

**Hybrid RRF Top Results**

1. Towards Interpreting BERT for Reading Comprehension Based QA
2. An initial attempt of combining visual selective attention with deep reinforcement learning
3. Only Train Once: A One-Shot Neural Network Training And Pruning Framework
4. PHS: A Toolbox for Parallel Hyperparameter Search
5. Typoglycemia under the Hood: Investigating Language Models' Understanding of Scrambled Words

### Query 2: Yann LeCun Convolutional Networks

**BM25 Top Results**

1. LSTM Fully Convolutional Networks for Time Series Classification
2. Multiscale Hierarchical Convolutional Networks
3. Multipod Convolutional Network
4. Striving for Simplicity: The All Convolutional Net
5. Detecting Exoplanet Transits through Machine Learning Techniques with Convolutional Neural Networks

**Vector Search Top Results**

1. An initial attempt of combining visual selective attention with deep reinforcement learning
2. Unified Implementations of Recurrent Neural Networks in Multiple Deep Learning Frameworks
3. Identifying Surgical Instruments in Pedagogical Cataract Surgery Videos through an Optimized Aggregation Network
4. Toward more frugal models for functional cerebral networks automatic recognition with resting-state fMRI
5. Reliable Identification of Redundant Kernels for Convolutional Neural Network Compression

**Hybrid RRF Top Results**

1. LSTM Fully Convolutional Networks for Time Series Classification
2. Toward more frugal models for functional cerebral networks automatic recognition with resting-state fMRI
3. An initial attempt of combining visual selective attention with deep reinforcement learning
4. Multiscale Hierarchical Convolutional Networks
5. Unified Implementations of Recurrent Neural Networks in Multiple Deep Learning Frameworks

### Query 3: Making Computers Understand Human Emotions from Text

**BM25 Top Results**

1. Emotion Concepts and their Function in a Large Language Model
2. SCoPE: Shift-Aware Speaker-Conditioned Priors for Emotion Recognition in Conversations
3. Benchmarking quantum computers via protocols
4. MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension
5. Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI Alignment in the Writing Process through Edits

**Vector Search Top Results**

1. Emotion Concepts and their Function in a Large Language Model
2. When Words Fail, Emojis Prevail: Generating Sarcastic Utterances with Emoji
3. Demystify Adult Learning: A Social Network and Large Language Model Assisted Approach
4. A Customer Journey in the Land of Oz: Leveraging the Wizard of Oz Technique to Model Emotions in Customer Service Interactions
5. Lightme: Analysing Language in Internet Support Groups for Mental Health

**Hybrid RRF Top Results**

1. Emotion Concepts and their Function in a Large Language Model
2. A Customer Journey in the Land of Oz: Leveraging the Wizard of Oz Technique to Model Emotions in Customer Service Interactions
3. SCoPE: Shift-Aware Speaker-Conditioned Priors for Emotion Recognition in Conversations
4. When Words Fail, Emojis Prevail: Generating Sarcastic Utterances with Emoji
5. Benchmarking quantum computers via protocols

## 5.4. Hybrid Search Analysis

### Which Method Performs Better?

There is no universally best method.

BM25 is particularly useful for:
- Exact technical phrases.
- Rare terminology.
- Acronyms.
- Names and identifiers.

Vector search is particularly useful for:
- Semantic similarity.
- Synonyms.
- Paraphrased queries.
- Conceptual descriptions.

Hybrid search combines the strengths of both approaches.

The actual results demonstrate that BM25 and vector retrieval can produce significantly different rankings.

### Can Hybrid Top-5 Contain Documents Absent from Both Individual Top-5 Lists?

Yes.

The implementation retrieves a wider candidate set from each method before applying RRF.

A document that appears below position five in individual result lists can still achieve a competitive combined RRF score.

This happens because RRF combines rankings rather than comparing raw BM25 and vector similarity scores.

### How Does Parameter k Affect RRF?

A larger `k` reduces the difference between high and lower ranks, producing smoother fusion.

A smaller `k` gives more weight to the highest-ranked documents.

The parameter should be evaluated using representative queries and relevance judgments.

---

# PART 6 — Analysis and Conclusions

## 6.1. Semantic Search vs Keyword Search

Semantic search is useful when a query describes an idea rather than repeating the exact words used in a document.

For example:

`making computers understand human emotions from text`

The query can retrieve documents about emotion recognition even when the exact query wording is absent from the paper.

Keyword search is often preferable for exact technical expressions, acronyms, identifiers, and names.

For example:

`BERT fine-tuning`

BM25 can strongly reward documents containing these exact terms.

Hybrid search is useful when a query contains both exact technical terminology and broader conceptual meaning.

**Conclusion:**

- Use semantic retrieval for conceptual questions and paraphrases.
- Use lexical retrieval for exact terms and names.
- Use hybrid retrieval when both signals are valuable.

## 6.2. How Does Chunk Size Affect Retrieval Quality?

Small chunks may lack sufficient context to represent a complete concept.

They can produce vague embeddings and increase the number of stored vectors.

Large chunks may combine multiple topics into a single embedding, reducing retrieval precision.

There is no universally optimal chunk size.

It depends on:
- Model token limits.
- Document structure.
- Query complexity.
- Desired retrieval granularity.
- Storage and processing constraints.

Chunk size and overlap should be evaluated using representative queries.

## 6.3. What If Pinecone Used Euclidean Distance for Normalized Vectors?

For normalized vectors:

\[
\|x\|_2=\|y\|_2=1
\]

The squared Euclidean distance is:

\[
\begin{aligned}
\|x-y\|_2^2
&=(x-y)\cdot(x-y)\\
&=\|x\|_2^2+\|y\|_2^2-2x\cdot y\\
&=1+1-2\operatorname{cos}(x,y)\\
&=2-2\operatorname{cos}(x,y)
\end{aligned}
\]

Therefore, minimizing Euclidean distance is mathematically equivalent to maximizing cosine similarity for unit vectors.

An Euclidean index would consequently return the same nearest-neighbor ordering as cosine for normalized vectors, subject to implementation precision and ties.

## 6.4. Pinecone Starter Limitations and Scaling to 10 Million Papers

A Starter plan is suitable for educational projects and smaller experiments.

However, free or starter plans may impose limits on:
- Index count.
- Storage.
- Vector count.
- Throughput.
- Available regions.
- Request sizes.
- Available resources.

Exact quotas and pricing depend on the current Pinecone plan and should be verified in the provider's documentation.

### Scaling to 10 Million Papers

For a production system containing 10 million scientific publications, I would consider the following:

1. Estimate vector storage and metadata requirements.
2. Benchmark query latency and retrieval quality.
3. Use batched and parallel ingestion.
4. Implement retry and backoff mechanisms.
5. Keep metadata compact.
6. Use incremental updates instead of rebuilding the complete index.
7. Store complete abstracts and documents outside the vector database.
8. Retrieve full documents by ID after vector search.
9. Monitor throughput, latency, and storage utilization.
10. Evaluate self-hosted Qdrant if infrastructure control or cost predictability becomes important.

For large-scale applications, vector search should be part of a broader retrieval architecture rather than the only storage layer.

---

# 7. Final Project Results

## 7.1. Execution Summary

All six scripts were executed successfully.

| Part | Implementation | Result |
|---|---|---|
| Part 1 | Dataset preparation and embeddings | Completed |
| Part 2 | Pinecone index and vector upload | Completed |
| Part 3 | Semantic search and metric comparison | Completed |
| Part 4 | Fixed-size and sentence-aware chunking | Completed |
| Part 5 | BM25 and hybrid RRF search | Completed |
| Part 6 | Theoretical analysis | Documented |

## 7.2. Key Metrics

| Metric | Result |
|---|---:|
| Source papers scanned | 3,261,651 |
| Papers selected | 10,000 |
| Embedding dimension | 768 |
| Embedding batches | 157 |
| Embedding generation time | 12 min 35 sec |
| Pinecone vectors uploaded | 10,000 |
| Pinecone upload batches | 50 |
| Pinecone upload time | 31 sec |
| Fixed-size chunks | 90 |
| Sentence-aware chunks | 62 |
| Hybrid search queries | 3 |

## 7.3. Main Findings

1. Reservoir Sampling provides an efficient way to select a representative sample from a large JSONL dataset without loading the entire dataset into memory.

2. L2 normalization makes cosine similarity and dot product mathematically equivalent.

3. For normalized vectors, Euclidean distance produces the same nearest-neighbor ranking as cosine similarity.

4. Metadata filtering can be combined with semantic search to narrow results by publication year and scientific category.

5. Chunking strategy affects the number of stored vectors and the search results.

6. Sentence-aware chunking produced approximately 31% fewer chunks than fixed-size chunking in this experiment.

7. BM25 and dense vector retrieval produce different rankings, demonstrating the complementary nature of lexical and semantic search.

8. Reciprocal Rank Fusion provides a practical way to combine multiple ranked result lists without comparing their raw scores.

---

# 8. Submission Checklist

- [x] Downloaded the arXiv dataset.
- [x] Prepared 10,000 records in Parquet format.
- [x] Generated 10,000 normalized embeddings.
- [x] Created the Pinecone `arxiv-papers` index.
- [x] Uploaded 10,000 vectors.
- [x] Executed pure semantic search.
- [x] Executed metadata-filtered searches.
- [x] Compared Cosine, Dot Product, and L2.
- [x] Implemented fixed-size chunking.
- [x] Implemented sentence-aware chunking.
- [x] Created separate Pinecone chunk indexes.
- [x] Executed chunk-based search experiments.
- [x] Implemented BM25 retrieval.
- [x] Implemented hybrid search with RRF.
- [x] Documented theoretical questions.
- [ ] Verify that `.env` and credentials are excluded from Git.
- [ ] Verify that downloaded data and generated embeddings are excluded from Git.
- [ ] Push the final repository to GitHub.
- [ ] Create and submit `Sakhno_nosql_2.zip`.

---

**Project Status: All six implementation scripts executed successfully.**