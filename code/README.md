# Improved TaxoGen

Improved implementation of TaxoGen (KDD 2018) using:

* SBERT semantic embeddings
* Global + Local semantic learning
* HDBSCAN clustering
* Recursive taxonomy generation
* Automatic evaluation metrics

---

# 1. Giới thiệu

Dự án này là phiên bản cải tiến của TaxoGen:

```text
TaxoGen:
Unsupervised Topic Taxonomy Construction
by Adaptive Term Embedding and Clustering
(KDD 2018)
```

Hệ thống tự động xây dựng cây taxonomy từ tập văn bản lớn.

Ví dụ:

```text
Artificial Intelligence
│
├── Neural Networks
│   ├── Deep Learning
│   ├── CNN
│   └── RNN
│
├── Support Vector Machines
│
└── Natural Language Processing
```

---

# 2. Cấu trúc project

```text
project/
│
├── config/
│   └── config.py
│
├── raw_pipeline/
│   └── download_data.py
│
├── processed_pipeline/
│   ├── finetune_sbert.py
│   ├── sbert_embedding.py
│   ├── build_taxonomy.py
│   └── local_sbert_training.py
│
├── evaluation/
│   ├── gen_eval.py
│   ├── eval.py
│   ├── compress_improved_taxonomy.py
│   ├── npmi_coherence.py
│   ├── silhouette_score.py
│   └── run_evaluation.py
│
├── data/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
└── main.py
```

---

# 3. Cài đặt

## Clone project

```bash
git clone YOUR_GITHUB_LINK
cd project
```

---

## Cài dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Dataset

Dataset KHÔNG được push lên GitHub vì dung lượng lớn.

Hệ thống hỗ trợ tự download dataset.

Chạy:

```bash
py raw_pipeline/download_data.py
```

---

# 5. Input tối thiểu

Pipeline hiện tại CHỈ cần:

```text
data/processed/
├── papers.txt
└── keywords.txt
```

---

# 6. Giải thích dữ liệu
link dữ liệu: https://drive.usercontent.google.com/download?id=1GbxKrxrmFrKt5vgDHP1xe1Qr_rfvR1jh&authuser=0
## 6.1 papers.txt

Đây là file QUAN TRỌNG NHẤT.

Mỗi dòng là một document đã được tiền xử lý.

Ví dụ:

```text
support vector machines are widely used for classification
deep neural networks improve image recognition
```

Ý nghĩa:

```text
1 dòng = 1 document
```

File này được dùng để:

* Global SBERT fine-tuning
* Local SBERT context
* Semantic learning
* Topic coherence
* Recursive taxonomy

---

## 6.2 keywords.txt

Danh sách keyword phrases dùng để xây taxonomy.

Ví dụ:

```text
support_vector_machine
deep_learning
graph_neural_network
```

File này được dùng để:

* sinh embeddings
* clustering
* build taxonomy node
* representative phrase ranking

---

# 7. Những file KHÔNG cần truyền vào

Pipeline sẽ tự sinh:

```text
index.txt
keyword_cnt.txt
doc_ids.txt
paper_cluster.txt
cluster_keywords.txt
```

Người dùng KHÔNG cần chuẩn bị các file này.

---

# 8. Luồng hoạt động

Toàn bộ pipeline:

```text
papers.txt
+
keywords.txt
↓
Global SBERT fine-tune
↓
Global embeddings
↓
Recursive clustering
↓
Local SBERT refinement
↓
Recursive taxonomy
↓
Taxonomy folders
↓
Evaluation
```

---

# 9. Chạy toàn bộ hệ thống

Chạy:

```bash
py main.py
```

Hệ thống sẽ tự:

```text
1. Fine-tune Global SBERT
2. Encode keyword embeddings
3. Recursive HDBSCAN clustering
4. Build local semantic context
5. Generate taxonomy tree
6. Export taxonomy
7. Save evaluation outputs
```

---

# 10. Taxonomy output

Taxonomy thật nằm tại:

```text
data/outputs/taxogen_tree/
```

Mỗi node là một thư mục.

Ví dụ:

```text
support_vector_machines/
├── keywords.txt
├── embeddings.txt
├── hierarchy.txt
├── caseolap.txt
└── ...
```

---

# 11. Taxonomy export

File export nằm tại:

```text
data/outputs/taxonomy/
```

Gồm:

```text
taxonomy.txt
taxonomy.json
compressed_taxonomy.txt
```

Các file này dùng cho:

* visualization
* evaluation
* human annotation

---

# 12. Compress taxonomy

Sinh compressed taxonomy:

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

Ví dụ:

```text
neural_networks/artificial_intelligence    machine_learning,deep_learning,...
```

---

# 13. Evaluation

## Sinh file đánh giá

```bash
py evaluation/gen_eval.py
```

---

## Chạy evaluation

```bash
py evaluation/run_evaluation.py
```

Metrics gồm:

* Topic Intrusion
* Parent-Child Relationship
* NPMI Coherence
* Silhouette Score

---

# 14. Global vs Local SBERT

## Global SBERT

Train trên toàn bộ corpus:

```text
papers.txt
```

Dùng cho:

```text
root taxonomy
(depth = 0)
```

---

## Local SBERT

Train semantic context riêng cho từng node con.

Ví dụ:

```text
Artificial Intelligence
→ chỉ dùng document thuộc AI
```

Dùng cho:

```text
depth > 0
```

Điều này giúp taxonomy semantic tốt hơn.

---

# 15. Storage Architecture

Code được chạy trên SSD.

Dataset + output + model + cache sẽ tự động chuyển sang ổ có dung lượng lớn hơn.

Ví dụ:

```text
SSD:
    source code

HDD:
    dataset
    outputs
    models
    cache
```

Hệ thống tự detect ổ lưu trữ khi chạy.

---

# 16. Output structure

Sau khi train:

```text
STORAGE_ROOT/
└── data/
    ├── raw/
    ├── processed/
    └── outputs/
        ├── embeddings/
        ├── models/
        ├── taxonomy/
        ├── taxogen_tree/
        └── logs/
```

---

# 17. GPU Support

Hệ thống tự detect GPU:

```text
cuda
```

hoặc:

```text
cpu
```

Nếu server có GPU, SBERT sẽ tự chạy bằng CUDA.

---

# 18. Cấu hình khuyến nghị

## Test local CPU

```python
EPOCHS = 1
MAX_TRAIN_PAIRS = 1000
BATCH_SIZE = 16
MAX_DEPTH = 2
```

---

## Train thật trên server GPU

```python
EPOCHS = 3
MAX_TRAIN_PAIRS = 350000
BATCH_SIZE = 128
MAX_DEPTH = 3
```

---

# 19. Workflow hoàn chỉnh

```text
Download dataset
↓
Fine-tune Global SBERT
↓
Generate embeddings
↓
Recursive clustering
↓
Local semantic refinement
↓
Taxonomy folders
↓
Compressed taxonomy
↓
Evaluation
```

---

# 20. Reference

TaxoGen:
Unsupervised Topic Taxonomy Construction
by Adaptive Term Embedding and Clustering

KDD 2018
