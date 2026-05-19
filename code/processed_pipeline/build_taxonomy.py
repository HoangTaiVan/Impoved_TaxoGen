import os
import sys
import re
import argparse
import numpy as np
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.utils import (
    read_lines,
    write_lines,
    ensure_parent_dir,
    cossim,
    load_txt_embeddings,
)

from common.taxonomy import Taxonomy, TNode

from config.config import (
    SBERT_MODEL,
    USE_LOCAL_SBERT,

    HDBSCAN_METRIC,
    CLUSTER_SELECTION_METHOD,
    CLUSTER_SELECTION_EPSILON,

    MAX_DEPTH,
    MIN_CLUSTER_SIZE,
    TOP_K,
)

# ======================================================
# OPTIONAL CONFIG VALUES
# Các tham số này nên đặt ở config.py để dễ chỉnh.
# Nếu config.py chưa có, code sẽ dùng default an toàn.
# ======================================================
try:
    from config.config import MIN_SAMPLES
except Exception:
    MIN_SAMPLES = None

try:
    from config.config import ROOT_N_CLUSTERS
except Exception:
    ROOT_N_CLUSTERS = 8

try:
    from config.config import ALPHA_BM25, BETA_COSINE
except Exception:
    ALPHA_BM25 = 0.4
    BETA_COSINE = 0.6

try:
    from config.config import N_CLUSTER_ITER, FILTER_RATIO, MIN_KEYWORDS_AFTER_FILTER
except Exception:
    N_CLUSTER_ITER = 2
    FILTER_RATIO = 0.75
    MIN_KEYWORDS_AFTER_FILTER = 30

from processed_pipeline.local_sbert_training import (
    build_local_embeddings_for_node,
)


# ======================================================
# GLOBAL SBERT MODEL
# ======================================================

_SBERT_MODEL = None


def get_sbert_model():
    global _SBERT_MODEL

    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print(f"[LOCAL SBERT] Loading model: {SBERT_MODEL}")
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL)

    return _SBERT_MODEL


# ======================================================
# BASIC WRITERS
# ======================================================

def write_doc_ids(doc_ids, path):
    write_lines([str(x) for x in doc_ids], path)


def write_seed_keywords(keywords, path):
    write_lines(keywords, path)


def save_node_text(doc_ids, documents, path):
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            if doc_id < 0 or doc_id >= len(documents):
                continue
            f.write(" ".join(documents[doc_id]) + "\n")


def save_node_embeddings(keywords, embeddings, path):
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        if embeddings is None or len(embeddings) == 0:
            f.write("0 0\n")
            return

        dim = embeddings.shape[1]
        f.write(f"{len(keywords)} {dim}\n")

        for kw, vec in zip(keywords, embeddings):
            vec_str = " ".join([str(float(x)) for x in vec])
            f.write(f"{kw} {vec_str}\n")


def save_score_files(scores, score_file, caseolap_file):
    ensure_parent_dir(score_file)
    ensure_parent_dir(caseolap_file)

    with open(score_file, "w", encoding="utf-8") as sf:
        for kw, score in scores:
            sf.write(f"{kw}\t{score:.6f}\n")

    with open(caseolap_file, "w", encoding="utf-8") as cf:
        for kw, score in scores:
            cf.write(f"{kw}\t{score:.6f}\n")


# ======================================================
# LABELING / RANKING: BM25 + COSINE
# ======================================================

def make_safe_node_name(name):
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("*", "root")
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def minmax_normalize(score_map):
    if not score_map:
        return {}

    values = list(score_map.values())
    mn = min(values)
    mx = max(values)

    if mx == mn:
        return {k: 1.0 for k in score_map}

    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def build_node_keyword_stats(doc_ids, documents):
    """
    Tính TF/DF trong phạm vi node hiện tại.
    Dùng cho BM25 thay cho CaseOLAP gốc.
    """
    keyword_tf = Counter()
    keyword_df = Counter()
    doc_lens = []

    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(documents):
            continue

        doc = documents[doc_id]
        if not doc:
            continue

        keyword_tf.update(doc)
        keyword_df.update(set(doc))
        doc_lens.append(len(doc))

    n_docs = len(doc_lens)
    avg_doc_len = float(np.mean(doc_lens)) if doc_lens else 0.0

    return keyword_tf, keyword_df, n_docs, avg_doc_len


def bm25_keyword_score(
    keyword,
    keyword_tf,
    keyword_df,
    n_docs,
    avg_doc_len,
    k1=1.5,
    b=0.75,
):
    tf = keyword_tf.get(keyword, 0)
    df = keyword_df.get(keyword, 0)

    if tf <= 0 or df <= 0 or n_docs <= 0 or avg_doc_len <= 0:
        return 0.0

    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # Dùng TF tổng trong node, chuẩn hóa đơn giản bằng avg length.
    denom = tf + k1 * (1.0 - b + b * 1.0)

    if denom <= 0:
        return 0.0

    return float(idf * ((tf * (k1 + 1.0)) / denom))


def rank_cluster_keywords(
    cluster_keywords,
    keyword_to_id,
    embeddings,
    top_k=10,
    doc_ids=None,
    documents=None,
    node_stats=None,
    alpha=ALPHA_BM25,
    beta=BETA_COSINE,
    return_scores=False,
):
    """
    Representative ranking = BM25 + Cosine.
    - Cosine: keyword gần centroid cluster.
    - BM25: keyword quan trọng trong documents của node.
    """
    valid_keywords = [kw for kw in cluster_keywords if kw in keyword_to_id]

    if len(valid_keywords) == 0:
        if return_scores:
            return []
        return cluster_keywords[:top_k]

    vecs = np.array([embeddings[keyword_to_id[kw]] for kw in valid_keywords])
    centroid = np.mean(vecs, axis=0)

    cosine_scores = {
        kw: cossim(embeddings[keyword_to_id[kw]], centroid)
        for kw in valid_keywords
    }

    # Nếu không có context document thì fallback cosine-only.
    if node_stats is None:
        if doc_ids is not None and documents is not None:
            node_stats = build_node_keyword_stats(doc_ids, documents)
        else:
            ranked = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
            if return_scores:
                return ranked[:top_k]
            return [kw for kw, _ in ranked[:top_k]]

    keyword_tf, keyword_df, n_docs, avg_doc_len = node_stats

    bm25_scores = {
        kw: bm25_keyword_score(
            keyword=kw,
            keyword_tf=keyword_tf,
            keyword_df=keyword_df,
            n_docs=n_docs,
            avg_doc_len=avg_doc_len,
        )
        for kw in valid_keywords
    }

    bm25_norm = minmax_normalize(bm25_scores)
    cosine_norm = minmax_normalize(cosine_scores)

    hybrid_scores = {
        kw: alpha * bm25_norm.get(kw, 0.0) + beta * cosine_norm.get(kw, 0.0)
        for kw in valid_keywords
    }

    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    if return_scores:
        return ranked[:top_k]

    return [kw for kw, _ in ranked[:top_k]]


def choose_node_label(
    cluster_id,
    cluster_keywords,
    keyword_to_id,
    embeddings,
    used_labels,
    top_k=10,
    doc_ids=None,
    documents=None,
    node_stats=None,
):
    representative = rank_cluster_keywords(
        cluster_keywords=cluster_keywords,
        keyword_to_id=keyword_to_id,
        embeddings=embeddings,
        top_k=top_k,
        doc_ids=doc_ids,
        documents=documents,
        node_stats=node_stats,
    )

    for kw in representative:
        label = make_safe_node_name(kw)
        if label and label not in used_labels:
            used_labels.add(label)
            return label, representative

    fallback = f"cluster_{cluster_id}"
    idx = 1
    while fallback in used_labels:
        fallback = f"cluster_{cluster_id}_{idx}"
        idx += 1

    used_labels.add(fallback)
    return fallback, representative


# ======================================================
# CLUSTERING: ROOT KMEANS / CHILD HDBSCAN
# ======================================================

def cluster_keywords_kmeans(keywords, embeddings, n_clusters=8):
    """
    Root dùng KMeans trên vector đã normalize.
    Gần với tinh thần Spherical KMeans của TaxoGen gốc.
    """
    if len(keywords) < n_clusters:
        return None

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    x = normalize(embeddings)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(x)

    print(
        f"[KMEANS] Root clustering: "
        f"{len(keywords)} keywords -> {n_clusters} clusters"
    )

    return labels


def reassign_noise_to_nearest_cluster(labels, embeddings):
    """
    HDBSCAN sinh noise = -1.
    Gán noise về centroid gần nhất để không mất quá nhiều keyword.
    """
    labels = np.array(labels, dtype=int)

    valid_labels = sorted([int(x) for x in set(labels) if int(x) != -1])
    if len(valid_labels) == 0:
        return labels

    centroids = {}
    for label in valid_labels:
        members = embeddings[labels == label]
        if len(members) > 0:
            centroids[label] = np.mean(members, axis=0)

    noise_indices = np.where(labels == -1)[0]

    for idx in noise_indices:
        vec = embeddings[idx]
        best_label = None
        best_score = -1e18

        for label, centroid in centroids.items():
            score = cossim(vec, centroid)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None:
            labels[idx] = best_label

    if len(noise_indices) > 0:
        print(f"[HDBSCAN] Reassigned noise keywords: {len(noise_indices)}")

    return labels


def cluster_keywords_hdbscan(keywords, embeddings, min_cluster_size=MIN_CLUSTER_SIZE):
    import hdbscan

    if len(keywords) < min_cluster_size * 2:
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    )

    raw_labels = clusterer.fit_predict(embeddings)
    valid = [x for x in set(raw_labels) if x != -1]

    if len(valid) == 0:
        return None

    labels = reassign_noise_to_nearest_cluster(raw_labels, embeddings)
    return labels


def cluster_keywords_auto(
    keywords,
    embeddings,
    depth,
    min_cluster_size=MIN_CLUSTER_SIZE,
):
    """
    Hybrid clustering:
    - depth = 0: KMeans để root có k nhánh lớn ổn định.
    - depth > 0: HDBSCAN để tách subtopic tự nhiên.
    """
    if depth == 0:
        return cluster_keywords_kmeans(
            keywords=keywords,
            embeddings=embeddings,
            n_clusters=ROOT_N_CLUSTERS,
        )

    return cluster_keywords_hdbscan(
        keywords=keywords,
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
    )


# ======================================================
# ADAPTIVE FILTERING
# ======================================================

def adaptive_filter_keywords(
    valid_keywords,
    node_embeddings,
    labels,
    doc_ids,
    documents,
    top_keep_ratio=FILTER_RATIO,
    min_keywords_after_filter=MIN_KEYWORDS_AFTER_FILTER,
    node_stats=None,
):
    """
    Adaptive Filtering:
    cluster -> BM25+Cosine rank -> filter -> recluster.
    Đây là phần thay thế tinh thần CaseOLAP filtering trong TaxoGen gốc.
    """
    cluster_map = defaultdict(list)

    for kw, label in zip(valid_keywords, labels):
        label = int(label)
        if label == -1:
            continue
        cluster_map[label].append(kw)

    if len(cluster_map) == 0:
        return valid_keywords, node_embeddings

    node_keyword_to_id = {kw: idx for idx, kw in enumerate(valid_keywords)}
    kept_keywords = []

    for _, cluster_keywords in cluster_map.items():
        keep_n = max(
            min_keywords_after_filter,
            int(len(cluster_keywords) * top_keep_ratio),
        )
        keep_n = min(keep_n, len(cluster_keywords))

        ranked = rank_cluster_keywords(
            cluster_keywords=cluster_keywords,
            keyword_to_id=node_keyword_to_id,
            embeddings=node_embeddings,
            top_k=keep_n,
            doc_ids=doc_ids,
            documents=documents,
            node_stats=node_stats,
        )

        kept_keywords.extend(ranked)

    kept_keywords = list(dict.fromkeys(kept_keywords))

    # Không lọc nếu kết quả quá ít.
    if len(kept_keywords) < min_keywords_after_filter:
        return valid_keywords, node_embeddings

    kept_set = set(kept_keywords)
    new_keywords = []
    new_embeddings = []

    for kw, emb in zip(valid_keywords, node_embeddings):
        if kw in kept_set:
            new_keywords.append(kw)
            new_embeddings.append(emb)

    return new_keywords, np.array(new_embeddings)


# ======================================================
# DOCUMENT MEMBERSHIP
# ======================================================

def assign_docs_to_clusters(doc_ids, documents, keywords, labels):
    keyword_cluster = {
        kw: int(label)
        for kw, label in zip(keywords, labels)
        if int(label) != -1
    }

    valid_clusters = sorted(set(keyword_cluster.values()))
    cluster_docs = defaultdict(list)

    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(documents):
            continue

        doc = documents[doc_id]
        score = {cid: 0 for cid in valid_clusters}

        for token in doc:
            if token in keyword_cluster:
                score[keyword_cluster[token]] += 1

        if not score:
            continue

        best_cluster = max(score, key=score.get)
        if score[best_cluster] > 0:
            cluster_docs[best_cluster].append(doc_id)

    return cluster_docs


# ======================================================
# EMBEDDING SELECTION: GLOBAL / LOCAL
# ======================================================

def get_node_embeddings(
    depth,
    doc_ids,
    seed_keywords,
    documents,
    keyword_to_id,
    global_embeddings,
    min_cluster_size,
):
    valid_keywords = [kw for kw in seed_keywords if kw in keyword_to_id]

    if len(valid_keywords) < min_cluster_size * 2:
        return [], np.array([])

    # Root node -> global SBERT
    if depth == 0 or not USE_LOCAL_SBERT:
        node_embeddings = np.array([
            global_embeddings[keyword_to_id[kw]]
            for kw in valid_keywords
        ])

        print("[EMBEDDING] Using GLOBAL SBERT")
        return valid_keywords, node_embeddings

    # Child node -> local SBERT
    print("[EMBEDDING] Using LOCAL SBERT")

    sbert_model = get_sbert_model()

    local_keywords, local_embeddings = build_local_embeddings_for_node(
        seed_keywords=seed_keywords,
        keyword_to_id=keyword_to_id,
        global_embeddings=global_embeddings,
        sbert_model=sbert_model,
        doc_ids=doc_ids,
        documents=documents,
    )

    return local_keywords, local_embeddings


# ======================================================
# RECURSIVE TAXOGEN BUILD
# ======================================================

def recursive_build(
    node_dir,
    node_name,
    doc_ids,
    seed_keywords,
    documents,
    all_keywords,
    keyword_to_id,
    embeddings,
    taxonomy,
    depth,
    max_depth,
    min_cluster_size,
    top_k,
):
    os.makedirs(node_dir, exist_ok=True)

    print(
        f"[NODE] depth={depth} node={node_name} "
        f"docs={len(doc_ids)} keywords={len(seed_keywords)}"
    )

    # ======================================================
    # TAXOGEN-STYLE NODE FILES: BEFORE SPLIT
    # ======================================================
    write_doc_ids(doc_ids, os.path.join(node_dir, "doc_ids.txt"))
    write_seed_keywords(seed_keywords, os.path.join(node_dir, "seed_keywords.txt"))
    write_lines(seed_keywords, os.path.join(node_dir, "keywords.txt"))
    save_node_text(doc_ids=doc_ids, documents=documents, path=os.path.join(node_dir, "text"))

    # ======================================================
    # STOP CONDITIONS
    # ======================================================
    if depth >= max_depth:
        print(f"[STOP] {node_name}: reach max_depth")
        save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
        save_node_embeddings([], np.array([]), os.path.join(node_dir, "embeddings.txt"))
        return

    if len(seed_keywords) < min_cluster_size * 2:
        print(f"[STOP] {node_name}: too few keywords")
        save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
        save_node_embeddings([], np.array([]), os.path.join(node_dir, "embeddings.txt"))
        return

    if len(doc_ids) < 20:
        print(f"[STOP] {node_name}: too few documents")
        save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
        save_node_embeddings([], np.array([]), os.path.join(node_dir, "embeddings.txt"))
        return

    # ======================================================
    # GLOBAL / LOCAL EMBEDDING
    # ======================================================
    valid_keywords, node_embeddings = get_node_embeddings(
        depth=depth,
        doc_ids=doc_ids,
        seed_keywords=seed_keywords,
        documents=documents,
        keyword_to_id=keyword_to_id,
        global_embeddings=embeddings,
        min_cluster_size=min_cluster_size,
    )

    if len(valid_keywords) < min_cluster_size * 2:
        print(f"[STOP] {node_name}: too few valid keywords")
        save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
        save_node_embeddings(valid_keywords, node_embeddings, os.path.join(node_dir, "embeddings.txt"))
        return

    # Tính node stats một lần để dùng cho BM25 trong adaptive + labeling.
    node_stats = build_node_keyword_stats(doc_ids, documents)

    # ======================================================
    # ADAPTIVE CLUSTERING LOOP
    # cluster -> BM25+Cosine filter -> recluster
    # ======================================================
    labels = None

    for iter_id in range(N_CLUSTER_ITER):
        print(
            f"[ADAPTIVE] node={node_name} "
            f"iter={iter_id + 1}/{N_CLUSTER_ITER} "
            f"keywords={len(valid_keywords)}"
        )

        labels = cluster_keywords_auto(
            keywords=valid_keywords,
            embeddings=node_embeddings,
            depth=depth,
            min_cluster_size=min_cluster_size,
        )

        if labels is None:
            print(f"[STOP] {node_name}: clustering failed in adaptive loop")
            save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
            return

        # Vòng cuối không lọc nữa; labels cuối dùng để tạo child.
        if iter_id < N_CLUSTER_ITER - 1:
            old_n = len(valid_keywords)

            valid_keywords, node_embeddings = adaptive_filter_keywords(
                valid_keywords=valid_keywords,
                node_embeddings=node_embeddings,
                labels=labels,
                doc_ids=doc_ids,
                documents=documents,
                top_keep_ratio=FILTER_RATIO,
                min_keywords_after_filter=MIN_KEYWORDS_AFTER_FILTER,
                node_stats=node_stats,
            )

            print(f"[ADAPTIVE] filtered keywords: {old_n} -> {len(valid_keywords)}")

            if len(valid_keywords) < min_cluster_size * 2:
                print(f"[STOP] {node_name}: too few keywords after adaptive filtering")
                save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
                save_node_embeddings(valid_keywords, node_embeddings, os.path.join(node_dir, "embeddings.txt"))
                return

    # Ghi lại keywords/embeddings sau adaptive filtering.
    write_lines(valid_keywords, os.path.join(node_dir, "keywords.txt"))
    save_node_embeddings(valid_keywords, node_embeddings, os.path.join(node_dir, "embeddings.txt"))

    # ======================================================
    # BUILD CLUSTER MAP FROM FINAL LABELS
    # ======================================================
    cluster_map = defaultdict(list)

    for kw, label in zip(valid_keywords, labels):
        label = int(label)
        if label == -1:
            continue
        cluster_map[label].append(kw)

    if len(cluster_map) == 0:
        print(f"[STOP] {node_name}: all noise")
        save_score_files([], os.path.join(node_dir, "keywords.txt-score.txt"), os.path.join(node_dir, "caseolap.txt"))
        return

    node_keyword_to_id = {kw: idx for idx, kw in enumerate(valid_keywords)}

    # ======================================================
    # OUTPUT FILES
    # ======================================================
    hierarchy_file = os.path.join(node_dir, "hierarchy.txt")
    cluster_keyword_file = os.path.join(node_dir, "cluster_keywords.txt")
    paper_cluster_file = os.path.join(node_dir, "paper_cluster.txt")
    score_file = os.path.join(node_dir, "keywords.txt-score.txt")
    caseolap_file = os.path.join(node_dir, "caseolap.txt")

    cluster_docs = assign_docs_to_clusters(
        doc_ids=doc_ids,
        documents=documents,
        keywords=valid_keywords,
        labels=labels,
    )

    used_labels = set()
    all_scores = []

    with open(hierarchy_file, "w", encoding="utf-8") as hf, \
         open(cluster_keyword_file, "w", encoding="utf-8") as ckf, \
         open(paper_cluster_file, "w", encoding="utf-8") as pcf:

        for cluster_id in sorted(cluster_map.keys()):
            child_folder_name, representative = choose_node_label(
                cluster_id=cluster_id,
                cluster_keywords=cluster_map[cluster_id],
                keyword_to_id=node_keyword_to_id,
                embeddings=node_embeddings,
                used_labels=used_labels,
                top_k=top_k,
                doc_ids=doc_ids,
                documents=documents,
                node_stats=node_stats,
            )

            ranked_with_scores = rank_cluster_keywords(
                cluster_keywords=cluster_map[cluster_id],
                keyword_to_id=node_keyword_to_id,
                embeddings=node_embeddings,
                top_k=len(cluster_map[cluster_id]),
                doc_ids=doc_ids,
                documents=documents,
                node_stats=node_stats,
                return_scores=True,
            )

            for kw, score in ranked_with_scores:
                all_scores.append((kw, score))

            if node_name == "*":
                child_node_name = child_folder_name
            else:
                child_node_name = f"{node_name}/{child_folder_name}"

            # hierarchy.txt: child parent
            hf.write(f"{child_folder_name} {node_name}\n")

            # cluster_keywords.txt
            for kw in cluster_map[cluster_id]:
                ckf.write(f"{cluster_id}\t{kw}\n")

            # paper_cluster.txt
            child_doc_ids = cluster_docs.get(cluster_id, [])
            for doc_id in child_doc_ids:
                pcf.write(f"{doc_id}\t{cluster_id}\n")

            taxonomy.add_node(
                TNode(
                    name=child_node_name,
                    ph_list=representative,
                )
            )

            child_dir = os.path.join(node_dir, child_folder_name)

            recursive_build(
                node_dir=child_dir,
                node_name=child_node_name,
                doc_ids=child_doc_ids,
                seed_keywords=cluster_map[cluster_id],
                documents=documents,
                all_keywords=all_keywords,
                keyword_to_id=keyword_to_id,
                embeddings=embeddings,
                taxonomy=taxonomy,
                depth=depth + 1,
                max_depth=max_depth,
                min_cluster_size=min_cluster_size,
                top_k=top_k,
            )

    all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    save_score_files(all_scores, score_file, caseolap_file)


# ======================================================
# ENTRY POINT
# ======================================================

def build_taxogen_style_tree(
    document_file,
    keyword_file,
    embedding_file,
    output_tree_dir,
    output_taxonomy_txt,
    output_taxonomy_json,
    max_depth=MAX_DEPTH,
    min_cluster_size=MIN_CLUSTER_SIZE,
    top_k=TOP_K,
):
    documents = [line.split() for line in read_lines(document_file)]
    keywords = read_lines(keyword_file)
    embedding_map = load_txt_embeddings(embedding_file)

    keywords = [kw for kw in keywords if kw in embedding_map]
    embeddings = np.array([embedding_map[kw] for kw in keywords])
    keyword_to_id = {kw: idx for idx, kw in enumerate(keywords)}

    if len(keywords) != len(embeddings):
        raise ValueError("keywords.txt và phrase_embeddings.txt không cùng số lượng.")

    all_doc_ids = list(range(len(documents)))

    os.makedirs(output_tree_dir, exist_ok=True)
    ensure_parent_dir(output_taxonomy_txt)
    ensure_parent_dir(output_taxonomy_json)

    taxonomy = Taxonomy()

    recursive_build(
        node_dir=output_tree_dir,
        node_name="*",
        doc_ids=all_doc_ids,
        seed_keywords=keywords,
        documents=documents,
        all_keywords=keywords,
        keyword_to_id=keyword_to_id,
        embeddings=embeddings,
        taxonomy=taxonomy,
        depth=0,
        max_depth=max_depth,
        min_cluster_size=min_cluster_size,
        top_k=top_k,
    )

    taxonomy.export_txt(output_taxonomy_txt)
    taxonomy.export_json(output_taxonomy_json)

    print("[DONE] TaxoGen-style tree saved:")
    print(f"  Tree folder : {output_tree_dir}")
    print(f"  Taxonomy txt: {output_taxonomy_txt}")
    print(f"  Taxonomy json: {output_taxonomy_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--document_file", required=True)
    parser.add_argument("--keyword_file", required=True)
    parser.add_argument("--embedding_file", required=True)
    parser.add_argument("--output_tree_dir", required=True)
    parser.add_argument("--output_taxonomy_txt", required=True)
    parser.add_argument("--output_taxonomy_json", required=True)

    parser.add_argument("--max_depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--min_cluster_size", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--top_k", type=int, default=TOP_K)

    args = parser.parse_args()

    build_taxogen_style_tree(
        document_file=args.document_file,
        keyword_file=args.keyword_file,
        embedding_file=args.embedding_file,
        output_tree_dir=args.output_tree_dir,
        output_taxonomy_txt=args.output_taxonomy_txt,
        output_taxonomy_json=args.output_taxonomy_json,
        max_depth=args.max_depth,
        min_cluster_size=args.min_cluster_size,
        top_k=args.top_k,
    )
