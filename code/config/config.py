import os
import shutil


# ======================================================
# ROOT PROJECT - CODE LOCATION
# ======================================================
# Thư mục chứa source code, thường nằm trên SSD.

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ======================================================
# AUTO DETECT STORAGE ROOT
# ======================================================
# Mục tiêu:
# - Code nằm ở SSD
# - Data, outputs, model, cache nằm ở ổ khác có dung lượng lớn hơn
#
# Windows:
#   tự quét các ổ D:\ E:\ F:\ ...
#
# Linux:
#   ưu tiên /mnt/hdd, /mnt/data, /data, /workspace/data
#
# Nếu không tìm thấy ổ khác:
#   fallback về ROOT_DIR/storage


def detect_storage_root():
    candidates = []

    # ======================================================
    # WINDOWS DRIVE DETECTION
    # ======================================================

    root_drive = os.path.splitdrive(ROOT_DIR)[0].upper()

    for drive in "DEFGHIJKLMNOPQRSTUVWXYZ":
        drive_path = f"{drive}:\\"

        if not os.path.exists(drive_path):
            continue

        drive_name = os.path.splitdrive(drive_path)[0].upper()

        # Tránh dùng cùng ổ với code nếu có ổ khác
        if drive_name == root_drive:
            continue

        try:
            total, used, free = shutil.disk_usage(drive_path)
            candidates.append((free, drive_path))
        except Exception:
            pass

    if len(candidates) > 0:
        candidates.sort(reverse=True)
        best_drive = candidates[0][1]

        return os.path.join(
            best_drive,
            "improved_taxogen_storage",
        )

    # ======================================================
    # LINUX SERVER DETECTION
    # ======================================================

    linux_candidates = [
        "/mnt/hdd",
        "/mnt/data",
        "/data",
        "/workspace/data",
    ]

    linux_valid = []

    for path in linux_candidates:
        if os.path.exists(path):
            try:
                total, used, free = shutil.disk_usage(path)
                linux_valid.append((free, path))
            except Exception:
                pass

    if len(linux_valid) > 0:
        linux_valid.sort(reverse=True)
        best_path = linux_valid[0][1]

        return os.path.join(
            best_path,
            "improved_taxogen_storage",
        )

    # ======================================================
    # FALLBACK
    # ======================================================

    return os.path.join(
        ROOT_DIR,
        "storage",
    )


STORAGE_ROOT = detect_storage_root()


# ======================================================
# DATA LOCATION - STORAGE DRIVE
# ======================================================
# Toàn bộ data nằm trên ổ lưu trữ, không nằm cạnh code.

DATA_DIR = os.path.join(STORAGE_ROOT, "data")

RAW_DIR = os.path.join(DATA_DIR, "raw")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


# ======================================================
# DATA DOWNLOAD
# ======================================================

DATASET_NAME = "dblp"

DATASET_URL = (
    "https://drive.google.com/uc?id=1GbxKrxrmFrKt5vgDHP1xe1Qr_rfvR1jh"
)

DATASET_ZIP_NAME = "dblp.zip"

# Alias để tương thích code cũ
DATA_URL = DATASET_URL
DATA_ZIP_NAME = DATASET_ZIP_NAME


# ======================================================
# INPUT FILES
# ======================================================

PAPERS_FILE = os.path.join(PROCESSED_DIR, "papers.txt")

KEYWORDS_FILE = os.path.join(PROCESSED_DIR, "keywords.txt")

INDEX_FILE = os.path.join(PROCESSED_DIR, "index.txt")

KEYWORD_CNT_FILE = os.path.join(PROCESSED_DIR, "keyword_cnt.txt")


# ======================================================
# OUTPUT DIRS - INSIDE DATA FOLDER
# ======================================================
# Theo yêu cầu:
#   output sinh trong DATA_DIR
#
# Cấu trúc:
#   STORAGE_ROOT/
#   └── data/
#       ├── raw/
#       ├── processed/
#       └── outputs/

OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")

CLUSTER_DIR = os.path.join(OUTPUT_DIR, "clusters")

TREE_DIR = os.path.join(OUTPUT_DIR, "taxogen_tree")

TAXONOMY_DIR = os.path.join(OUTPUT_DIR, "taxonomy")

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


# ======================================================
# OUTPUT FILES
# ======================================================

EMBEDDING_FILE = os.path.join(
    EMBEDDING_DIR,
    "phrase_embeddings.txt",
)

GLOBAL_CLUSTER_FILE = os.path.join(
    CLUSTER_DIR,
    "hdbscan_labels.tsv",
)

GLOBAL_RANK_FILE = os.path.join(
    CLUSTER_DIR,
    "ranked_keywords.tsv",
)

TAXONOMY_TXT = os.path.join(
    TAXONOMY_DIR,
    "taxonomy.txt",
)

TAXONOMY_JSON = os.path.join(
    TAXONOMY_DIR,
    "taxonomy.json",
)


# ======================================================
# SBERT GLOBAL ENCODING
# ======================================================

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Server mạnh có thể đổi:
# SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

EMBEDDING_FORMAT = "txt"

NORMALIZE_EMBEDDINGS = True


# ======================================================
# GLOBAL SBERT FINE-TUNING
# ======================================================

USE_FINE_TUNE = True

BASE_SBERT_MODEL = SBERT_MODEL

FINETUNED_MODEL_DIR = os.path.join(
    MODEL_DIR,
    "sbert_dblp_finetuned",
    # "sbert_amazon_electronics_finetuned",
)

# Không nên để 100 hoặc 500 epoch.
# Với SBERT, 2-3 epoch thường hợp lý.
EPOCHS = 3

BATCH_SIZE = 64

LEARNING_RATE = 2e-5

WARMUP_RATIO = 0.1

MAX_TRAIN_PAIRS = 200000

PAIR_WINDOW_SIZE = 5


# ======================================================
# ROOT CLUSTERING - KMEANS
# ======================================================
# Root dùng KMeans để chia các nhánh lớn tương đối cân bằng,
# gần với tinh thần Spherical KMeans của TaxoGen gốc.

ROOT_N_CLUSTERS = 6


# ======================================================
# CHILD CLUSTERING - HDBSCAN
# ======================================================
# Các node con dùng HDBSCAN để tách sub-topic tự nhiên và xử lý noise.

MIN_CLUSTER_SIZE = 10

MIN_SAMPLES = 1

CLUSTER_SELECTION_METHOD = "eom"

CLUSTER_SELECTION_EPSILON = 0.20

HDBSCAN_METRIC = "euclidean"


# ======================================================
# TAXONOMY BUILDING
# ======================================================

MAX_DEPTH = 3

TOP_K = 10

MIN_DOCS_TO_SPLIT = 20

MIN_KEYWORDS_TO_SPLIT = MIN_CLUSTER_SIZE * 2


# ======================================================
# ADAPTIVE CLUSTERING LOOP
# ======================================================
# Giống tinh thần TaxoGen gốc:
# cluster -> rank/filter -> recluster.

N_CLUSTER_ITER = 2

# Giữ lại 75% keyword tốt nhất trong mỗi cluster sau BM25+Cosine ranking.
FILTER_RATIO = 0.75

# Không lọc quá mạnh với node nhỏ.
MIN_KEYWORDS_AFTER_FILTER = 30


# ======================================================
# BM25 + COSINE RANKING
# ======================================================

ALPHA_BM25 = 0.4

BETA_COSINE = 0.6


# ======================================================
# LOCAL SBERT
# ======================================================

USE_LOCAL_SBERT = True

LOCAL_CONTEXT_DOCS = 300

LAMBDA_KEYWORD = 0.7


# ======================================================
# LOCAL SBERT FINE-TUNING
# ======================================================
# Bật True trên server nếu muốn gần TaxoGen paper hơn.
# Lưu ý: local fine-tune rất tốn thời gian vì chạy ở từng node.

USE_LOCAL_FINE_TUNE = True

LOCAL_FINE_TUNE_EPOCHS = 2

LOCAL_FINE_TUNE_BATCH_SIZE = 64

LOCAL_FINE_TUNE_LR = 2e-5

LOCAL_MAX_TRAIN_PAIRS = 10000

LOCAL_MIN_TRAIN_PAIRS = 100


# ======================================================
# SERVER / CACHE
# ======================================================
# Cache HuggingFace cũng để trên ổ lưu trữ.

HF_HOME = os.path.join(
    STORAGE_ROOT,
    "hf_cache",
)

NUM_WORKERS = 8

PIN_MEMORY = True


# ======================================================
# AUTO DEVICE DETECTION
# ======================================================

try:
    import torch

    if torch.cuda.is_available():
        DEVICE = "cuda"
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        DEVICE = "cpu"
        GPU_NAME = "CPU"

except Exception:
    DEVICE = "cpu"
    GPU_NAME = "CPU"


if DEVICE == "cpu":
    PIN_MEMORY = False


# ======================================================
# RANDOM
# ======================================================

RANDOM_SEED = 42


# ======================================================
# DEBUG PRINT
# ======================================================

print(f"[ROOT_DIR] {ROOT_DIR}")
print(f"[STORAGE_ROOT] {STORAGE_ROOT}")
print(f"[DATA_DIR] {DATA_DIR}")
print(f"[OUTPUT_DIR] {OUTPUT_DIR}")
print(f"[DEVICE] {DEVICE}")
print(f"[GPU] {GPU_NAME}")