# Improved TaxoGen: Semantic Taxonomy Construction with SBERT and HDBSCAN

An improved implementation of TaxoGen (KDD 2018) for unsupervised topic taxonomy construction using semantic representation learning, density-based clustering, and recursive taxonomy expansion.

The proposed framework replaces the original Word2Vec–CaseOLAP pipeline with a modern semantic architecture based on SBERT, HDBSCAN, BM25 ranking, and local semantic refinement, enabling the generation of higher-quality topic taxonomies from large-scale text corpora.

---

# Overview

Topic taxonomy construction aims to automatically organize concepts into a hierarchical structure.

For example:

```text
Artificial Intelligence
│
├── Machine Learning
│   ├── Deep Learning
│   ├── Convolutional Neural Networks
│   └── Recurrent Neural Networks
│
├── Support Vector Machines
│
└── Natural Language Processing
```

Such taxonomies are widely used in:

* Knowledge organization
* Information retrieval
* Semantic search
* Topic discovery
* Recommendation systems
* Scientific literature analysis

The original TaxoGen framework introduced an unsupervised taxonomy generation approach based on adaptive term embeddings and recursive clustering. However, it relies heavily on Word2Vec embeddings, K-Means clustering, and CaseOLAP ranking, which may limit semantic understanding and clustering flexibility in modern large-scale datasets.

To address these limitations, this project introduces a semantic-aware taxonomy generation framework based on SBERT and HDBSCAN.

---

# Architecture

The overall architecture of the proposed framework is illustrated below.
<img width="1495" height="1002" alt="Ảnh chụp màn hình 2026-06-07 231650" src="https://github.com/user-attachments/assets/b2b4a85c-524b-4810-8fe7-c7a8be05d344" />


# Main Contributions

Compared with the original TaxoGen, the proposed framework introduces four major improvements.

| Original TaxoGen | Improved TaxoGen         |
| ---------------- | ------------------------ |
| Word2Vec         | SBERT                    |
| Local Word2Vec   | Local SBERT              |
| K-Means          | HDBSCAN                  |
| CaseOLAP         | BM25 + Cosine Similarity |

## 1. Semantic-aware Embedding

The framework replaces Word2Vec with Sentence-BERT (SBERT) to generate contextual semantic representations for keyword phrases.

Unlike static word embeddings, SBERT captures semantic relationships among phrases and produces more meaningful representations for taxonomy construction.

---

## 2. Recursive Local Semantic Learning

After generating root-level clusters, each taxonomy node learns its own local semantic space.

A local SBERT model is refined using documents belonging to the corresponding topic, allowing the framework to distinguish fine-grained subtopics more effectively.

---

## 3. Adaptive Density-based Clustering

Instead of using K-Means at all hierarchy levels, the framework employs HDBSCAN for recursive taxonomy expansion.

Advantages include:

* Automatic cluster number discovery
* Noise detection
* Density-aware clustering
* Better handling of imbalanced topic structures

---

## 4. Semantic Phrase Ranking

Representative phrases are selected using a hybrid ranking strategy combining:

* BM25 relevance score
* Cosine similarity to cluster centroids

This allows the generated taxonomy nodes to be both statistically important and semantically representative.

---

# Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Supported Datasets

The framework currently supports:

## 1. DBLP Dataset

A preprocessed DBLP package is provided and can be used directly.

Download:

```bash
py raw_pipeline/download_data.py
```

After downloading, the required files are automatically prepared:

```text
data/processed/
├── papers.txt
├── keywords.txt
├── index.txt
└── keyword_cnt.txt
```

Dataset link:

```text
https://drive.usercontent.google.com/download?id=1GbxKrxrmFrKt5vgDHP1xe1Qr_rfvR1jh
```

---

## 2. Amazon Fashion 2023 Dataset

Official source:

https://amazon-reviews-2023.github.io/

Download raw Amazon Fashion data:

```bash
py raw_pipeline/download_amazon_fashion.py
```

The downloader retrieves:

```text
Amazon_Fashion.jsonl
meta_Amazon_Fashion.jsonl
```

and stores them under:

```text
data/raw/amazon_fashion/
```

---

## Amazon Fashion Preprocessing

Unlike DBLP, Amazon Fashion cannot be used directly.

The raw dataset must first be converted into TaxoGen input format.

Run:

```bash
py processed_pipeline/prepare_amazon_fashion.py
```

This preprocessing stage performs:

1. Metadata loading
2. Text cleaning
3. Product document construction
4. Phrase extraction
5. Vocabulary filtering
6. Keyword generation
7. papers.txt generation
8. keywords.txt generation

After preprocessing:

```text
data/processed/
├── papers.txt
└── keywords.txt
```

will be generated automatically.

These files are then used by the taxonomy generation pipeline.

---

# Required Input Files

The framework only requires:

```text
data/processed/
├── papers.txt
└── keywords.txt
```

---

## papers.txt

Each line represents one preprocessed document.

Example:

```text
support vector machines are widely used for classification
deep neural networks improve image recognition
```

Used for:

* Global SBERT fine-tuning
* Local semantic learning
* Topic coherence computation
* Recursive taxonomy generation

---

## keywords.txt

Contains candidate keyword phrases.

Example:

```text
support_vector_machine
deep_learning
graph_neural_network
```

Used for:

* Embedding generation
* Topic clustering
* Taxonomy node construction
* Representative phrase ranking

---

# Automatically Generated Files

The framework automatically generates:

```text
index.txt
keyword_cnt.txt
doc_ids.txt
paper_cluster.txt
cluster_keywords.txt
```

Users do not need to prepare these files.

---

# Running the Complete Pipeline

Run:

```bash
py main.py
```

The system automatically performs:

```text
1. Global SBERT fine-tuning
2. Keyword embedding generation
3. Root taxonomy construction
4. Recursive clustering
5. Local semantic refinement
6. Taxonomy generation
7. Taxonomy export
8. Automatic evaluation
```

---

# Taxonomy Output

Generated taxonomy nodes are stored in:

```text
data/outputs/taxogen_tree/
```

Example:

```text
support_vector_machines/
├── keywords.txt
├── embeddings.txt
├── hierarchy.txt
├── cluster_keywords.txt
└── ...
```

---

# Exported Taxonomy

Exported files are stored in:

```text
data/outputs/taxonomy/
```

Files include:

```text
taxonomy.txt
taxonomy.json
compressed_taxonomy.txt
```

These files can be used for:

* Visualization
* Human evaluation
* Downstream applications

---

# Taxonomy Compression

Generate a compressed taxonomy:

```bash
py evaluation/compress_improved_taxonomy.py ^
-root data/outputs/taxogen_tree ^
-output data/outputs/taxonomy/compressed_taxonomy.txt ^
-N 10
```

Format:

```text
taxonomy_path<TAB>representative_keywords
```

Example:

```text
machine_learning/deep_learning    cnn,rnn,transformer,...
```

---

# Evaluation

Generate evaluation files:

```bash
py evaluation/gen_eval.py
```

Run evaluation:

```bash
py evaluation/run_evaluation.py
```

Supported metrics:

* Fleiss’ Kappa
* Topic Intrusion
* Parent–Child Relationship

These metrics evaluate both taxonomy quality and annotator agreement.

---

# Global vs Local Semantic Learning

## Global SBERT

Trained on the entire corpus:

```text
papers.txt
```

Used for:

```text
Root-level taxonomy construction
```

---

## Local SBERT

Trained on documents associated with a specific taxonomy node.

Example:

```text
Artificial Intelligence
```

Only documents related to AI are used.

Used for:

```text
Depth > 0 taxonomy refinement
```

This enables the framework to learn topic-specific semantic structures.

---

# Storage Architecture

Source code remains on the system drive, while datasets, models, outputs, and cache files can be stored on a larger storage device.

Example:

```text
SSD
 └── source code

HDD
 ├── datasets
 ├── models
 ├── outputs
 └── cache
```

---

# GPU Support

The framework automatically detects:

```text
CUDA GPU
```

or

```text
CPU
```

SBERT training automatically uses GPU acceleration when available.

---

# Recommended Configuration

## Local CPU Testing

```python
EPOCHS = 1
MAX_TRAIN_PAIRS = 1000
BATCH_SIZE = 16
MAX_DEPTH = 2
```

---

## Full GPU Training

```python
EPOCHS = 3
MAX_TRAIN_PAIRS = 350000
BATCH_SIZE = 128
MAX_DEPTH = 3
```

---

# Workflow

## DBLP Workflow
<img width="425" height="579" alt="image" src="https://github.com/user-attachments/assets/d67d9f64-604d-48a5-8902-c738330fec7c" />

## Amazon Fashion Workflow
<img width="464" height="640" alt="image" src="https://github.com/user-attachments/assets/f2b45a13-1ca1-4937-80aa-146516c45920" />

# Reference

Zhang, C., Wang, H., Wang, J., et al.

**Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering.**

Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2018).
