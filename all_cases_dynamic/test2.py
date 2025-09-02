# import json
# import string
# import time
# import os
# import re
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from langchain.embeddings.base import Embeddings
# import numpy as np

# # --------------------- CONFIGURATION --------------------- #
# TOP_K = 20
# SEMANTIC_THRESHOLD = 0.6
# RRF_K = 60  # typical RRF constant (60 is common and robust)

# # --------------------- CUSTOM EMBEDDINGS CLASS --------------------- #
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(texts).tolist()

#     def embed_query(self, text):
#         return self.model.encode(text).tolist()

# # --------------------- DATA LOADING --------------------- #
# def load_data():
#     faq_path = r"Retrieval_evaluation/final.json"
#     test_path = r"all_cases_dynamic/test.json"

#     with open(faq_path, "r", encoding="utf-8") as f:
#         faq_data = json.load(f)
#     with open(test_path, "r", encoding="utf-8") as f:
#         test_queries = json.load(f)
#     return faq_data, test_queries

# # --------------------- VECTOR STORE --------------------- #
# def create_vector_store(embeddings, faq_data):
#     documents = [
#         Document(
#             page_content=item["question"],
#             metadata={
#                 "answer": item["answer"],
#                 "original_question": item["question"],
#                 "keywords": item.get("keywords", []),
#                 "index": idx
#             }
#         )
#         for idx, item in enumerate(faq_data)
#     ]
#     return FAISS.from_documents(documents, embeddings)

# # --------------------- BM25 INDEX --------------------- #
# def create_bm25_index(faq_data):
#     tokenized_questions = []
#     for item in faq_data:
#         text = " ".join([item["question"]] + item.get("keywords", []))
#         processed = text.lower().translate(str.maketrans('', '', string.punctuation))
#         tokenized_questions.append(processed.split())
#     return BM25Okapi(tokenized_questions)

# # --------------------- UTILITIES --------------------- #
# def minmax_norm(arr):
#     """Safely min-max normalize to [0,1]. Returns zeros if constant."""
#     a = np.array(arr, dtype=float)
#     mn = np.min(a)
#     mx = np.max(a)
#     denom = (mx - mn)
#     if denom <= 1e-12:
#         return np.zeros_like(a)
#     return (a - mn) / (denom + 1e-12)

# def softmax(x, temperature=1.0):
#     x = np.array(x, dtype=float) / max(1e-12, temperature)
#     x = x - np.max(x)
#     e = np.exp(x)
#     s = e.sum()
#     if s <= 1e-12:
#         return np.full_like(e, 1.0/len(e))
#     return e / s

# def semantic_results(query, vector_store, top_k=TOP_K):
#     """
#     Returns:
#       sem_list: list of (doc_index, sim_score in [0,1]) of top_k semantic results
#       sem_map: dict doc_index -> sim_score
#     """
#     # vector_store.similarity_search_with_score returns (doc, distance)
#     # Lower distance is better; map to similarity via 1/(1+dist)
#     res = vector_store.similarity_search_with_score(query, k=top_k)
#     pairs = []
#     for doc, dist in res:
#         idx = doc.metadata.get("index")
#         sim = 1.0 / (1.0 + float(dist))
#         pairs.append((idx, sim))
#     # Normalize similarities among retrieved set to [0,1] for stability
#     if pairs:
#         sims = [s for _, s in pairs]
#         sims_norm = minmax_norm(sims).tolist()
#         pairs = [(pairs[i][0], sims_norm[i]) for i in range(len(pairs))]
#     sem_map = {i: s for i, s in pairs}
#     return pairs, sem_map

# def bm25_scores_all(query, bm25_index, faq_data):
#     """
#     Returns:
#       scores (list length N), norm_scores (list length N in [0,1]),
#       rank_map (doc_index -> 1-based rank, ties by order)
#     """
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     scores = bm25_index.get_scores(processed)
#     norm_scores = minmax_norm(scores).tolist()

#     # ranks: higher score => better (rank 1)
#     order = sorted(range(len(scores)), key=lambda i: -scores[i])
#     rank_map = {doc_idx: r+1 for r, doc_idx in enumerate(order)}
#     return scores, norm_scores, rank_map

# def make_sem_rank_map(sem_pairs, corpus_size):
#     """
#     Build ranks for semantic using only the retrieved set; non-retrieved get large rank.
#     """
#     # higher sim => better (rank 1)
#     order = sorted(range(len(sem_pairs)), key=lambda i: -sem_pairs[i][1])
#     rank_map = {}
#     for r, pos in enumerate(order):
#         doc_idx = sem_pairs[pos][0]
#         rank_map[doc_idx] = r+1
#     # non-retrieved docs: assign large rank
#     default_rank = corpus_size + TOP_K + 1
#     return rank_map, default_rank

# def method_confidence_from_margin(scores):
#     # Convert to list if it's numpy array
#     if isinstance(scores, np.ndarray):
#         scores = scores.tolist()

#     # Handle empty case
#     if scores is None or len(scores) == 0:
#         return 0.0  

#     if len(scores) == 1:
#         return 1.0  

#     # Sort descending
#     sorted_scores = sorted(scores, reverse=True)
#     margin = sorted_scores[0] - sorted_scores[1]
#     return max(0.0, min(1.0, margin))


# # --------------------- RETRIEVAL METHODS --------------------- #
# def bm25_only(query, bm25_index, faq_data):
#     _, _, _ = None, None, None
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     scores = bm25_index.get_scores(processed)
#     top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
#     return [faq_data[i]["question"] for i in top_indices]

# def semantic_only(query, vector_store):
#     results = vector_store.similarity_search_with_score(query, k=TOP_K)
#     return [doc.metadata["original_question"] for doc, _ in results]

# def hybrid_rrf(query, vector_store, bm25_index, faq_data, rrf_k=RRF_K):
#     """
#     Reciprocal Rank Fusion:
#       score(d) = sum_m 1 / (rrf_k + rank_m(d))
#     Robust to scale differences, strong Recall@K in practice.
#     """
#     N = len(faq_data)

#     # BM25 ranks over full corpus
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     bm25_raw = bm25_index.get_scores(processed)
#     bm25_order = sorted(range(N), key=lambda i: -bm25_raw[i])
#     bm25_rank = {doc_idx: r+1 for r, doc_idx in enumerate(bm25_order)}

#     # Semantic ranks over top-K (others get large rank)
#     sem_pairs, _ = semantic_results(query, vector_store, top_k=TOP_K)
#     sem_rank, sem_default_rank = make_sem_rank_map(sem_pairs, N)

#     fused = []
#     for i in range(N):
#         r_b = bm25_rank.get(i, N + 1)
#         r_s = sem_rank.get(i, sem_default_rank)
#         score = 1.0 / (rrf_k + r_b) + 1.0 / (rrf_k + r_s)
#         fused.append((score, i))

#     top = sorted(fused, key=lambda x: -x[0])[:TOP_K]
#     return [faq_data[i]["question"] for _, i in top]

# def hybrid_adaptive_global(query, vector_store, bm25_index, faq_data):
#     """
#     Global adaptive weights per query:
#       1) Normalize BM25 across all docs.
#       2) Normalize semantic among retrieved set; zero elsewhere.
#       3) Compute method "confidence" by margin (top1 - top2).
#       4) Weights = softmax(confidences / temperature).
#       5) final_score = w_sem * sem_norm + w_bm25 * bm25_norm + optional keyword boost.
#     """
#     N = len(faq_data)

#     # BM25 across all docs
#     bm25_raw, bm25_norm, _ = bm25_scores_all(query, bm25_index, faq_data)

#     # Semantic top-K, normalize within retrieved
#     sem_pairs, sem_map = semantic_results(query, vector_store, top_k=TOP_K)

#     # Build per-doc normalized semantic vector (0 where not retrieved)
#     sem_norm = np.zeros(N, dtype=float)
#     for idx, s in sem_pairs:
#         sem_norm[idx] = s  # already min-max normalized among retrieved

#     # Compute confidences from margins (use normalized arrays)
#     sem_conf = method_confidence_from_margin(sem_norm)
#     bm25_conf = method_confidence_from_margin(bm25_norm)

#     # More decisive weighting via temperature (lower => sharper)
#     sem_w, bm25_w = softmax([sem_conf, bm25_conf], temperature=0.5)

#     # Keyword boost setup
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()

#     combined_scores = []
#     for i, item in enumerate(faq_data):
#         kb = 0.0
#         kws = item.get("keywords", [])
#         if kws:
#             kb = sum(1 for w in processed if w in kws) * 0.05

#         score = sem_w * sem_norm[i] + bm25_w * bm25_norm[i] + kb
#         combined_scores.append((score, i))

#     sorted_final = sorted(combined_scores, key=lambda x: -x[0])[:TOP_K]
#     return [faq_data[i]["question"] for _, i in sorted_final]

# # --------------------- EVALUATION --------------------- #
# def evaluate(method_fn, test_queries, *args):
#     metrics = {'top1_acc': 0, 'recall@3': 0, 'response_times': []}
#     for sample in test_queries:
#         start_time = time.time()
#         retrieved = method_fn(sample["query"], *args)
#         metrics['response_times'].append((time.time() - start_time) * 1000)

#         expected_list = sample["expected"] if isinstance(sample["expected"], list) else [sample["expected"]]
#         if retrieved and retrieved[0] in expected_list:
#             metrics['top1_acc'] += 1
#         if any(r in retrieved[:3] for r in expected_list):
#             metrics['recall@3'] += 1

#     total = len(test_queries)
#     metrics['top1_acc'] = metrics['top1_acc'] / total * 100
#     metrics['recall@3'] = metrics['recall@3'] / total * 100
#     metrics['avg_time'] = sum(metrics['response_times']) / total
#     return metrics

# # --------------------- LOGGING --------------------- #
# def log_results_to_json(results, hybrid_details, config, file_path="ll.json"):
#     log_entry = {
#         "TOP_K": config.get("TOP_K", TOP_K),
#         "SEMANTIC_THRESHOLD": config.get("SEMANTIC_THRESHOLD", SEMANTIC_THRESHOLD),
#         "RRF_K": config.get("RRF_K", RRF_K),
#         "total_queries": hybrid_details["total_queries"],
#         "results": results
#     }
#     if os.path.exists(file_path):
#         with open(file_path, "r", encoding="utf-8") as f:
#             history = json.load(f)
#     else:
#         history = []
#     history.append(log_entry)
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)

# # --------------------- MAIN EXECUTION --------------------- #
# if __name__ == "__main__":
#     faq_data, test_queries = load_data()
#     embeddings = SentenceTransformerEmbeddings()
#     vector_store = create_vector_store(embeddings, faq_data)
#     bm25_index = create_bm25_index(faq_data)

#     results = {}
#     methods = [
#         ("BM25 Only", bm25_only, [bm25_index, faq_data]),
#         ("Semantic Only", semantic_only, [vector_store]),
#         ("Hybrid (RRF)", hybrid_rrf, [vector_store, bm25_index, faq_data]),
#         ("Hybrid (Adaptive-Global)", hybrid_adaptive_global, [vector_store, bm25_index, faq_data]),
#     ]

#     for name, fn, args in methods:
#         results[name] = evaluate(fn, test_queries, *args)

#     # Print results
#     print("\nEvaluation Results:")
#     print("| Method                   | Top-1 Acc | Recall@3 | Avg Time (ms) |")
#     print("|--------------------------|-----------|----------|---------------|")
#     for name, metrics in results.items():
#         print(f"| {name:<24} | {metrics['top1_acc']:>8.2f}% | {metrics['recall@3']:>8.2f}% | {metrics['avg_time']:>13.2f} |")

#     # Choose one hybrid to log as "primary"
#     primary_hybrid_key = "Hybrid (RRF)"
#     hybrid_metrics = results[primary_hybrid_key]
#     config = {
#         "TOP_K": TOP_K,
#         "SEMANTIC_THRESHOLD": SEMANTIC_THRESHOLD,
#         "RRF_K": RRF_K
#     }
#     results[primary_hybrid_key]["total_queries"] = len(test_queries)
#     log_results_to_json(results, results[primary_hybrid_key], config)

#     print(f"\n{primary_hybrid_key} Details:")
#     print(f"- Top-1 Accuracy: {hybrid_metrics['top1_acc']:.2f}%")
#     print(f"- Recall@3: {hybrid_metrics['recall@3']:.2f}%")
#     print(f"- Average Response Time: {hybrid_metrics['avg_time']:.2f} ms")
#     print(f"- Total Queries Processed: {len(test_queries)}")
import json
import string
import time
import os
import re
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
import numpy as np

# --------------------- CONFIGURATION --------------------- #
TOP_K = 20
SEMANTIC_THRESHOLD = 0.6
RRF_K = 60  # typical RRF constant (60 is common and robust)

# --------------------- CUSTOM EMBEDDINGS CLASS --------------------- #
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# --------------------- DATA LOADING --------------------- #
def load_data():
    faq_path = r"Retrieval_evaluation/final.json"
    test_path = r"all_cases_dynamic/test.json"

    with open(faq_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    return faq_data, test_queries

# --------------------- VECTOR STORE --------------------- #
def create_vector_store(embeddings, faq_data):
    documents = [
        Document(
            page_content=item["question"],
            metadata={
                "answer": item["answer"],
                "original_question": item["question"],
                "keywords": item.get("keywords", []),
                "index": idx
            }
        )
        for idx, item in enumerate(faq_data)
    ]
    return FAISS.from_documents(documents, embeddings)

# --------------------- BM25 INDEX --------------------- #
def create_bm25_index(faq_data):
    tokenized_questions = []
    for item in faq_data:
        text = " ".join([item["question"]] + item.get("keywords", []))
        processed = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokenized_questions.append(processed.split())
    return BM25Okapi(tokenized_questions)

# --------------------- UTILITIES --------------------- #
def minmax_norm(arr):
    a = np.array(arr, dtype=float)
    mn = np.min(a)
    mx = np.max(a)
    denom = (mx - mn)
    if denom <= 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (denom + 1e-12)

def softmax(x, temperature=1.0):
    x = np.array(x, dtype=float) / max(1e-12, temperature)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 1e-12:
        return np.full_like(e, 1.0/len(e))
    return e / s

def semantic_results(query, vector_store, top_k=TOP_K):
    res = vector_store.similarity_search_with_score(query, k=top_k)
    pairs = []
    for doc, dist in res:
        idx = doc.metadata.get("index")
        sim = 1.0 / (1.0 + float(dist))
        pairs.append((idx, sim))
    if pairs:
        sims = [s for _, s in pairs]
        sims_norm = minmax_norm(sims).tolist()
        pairs = [(pairs[i][0], sims_norm[i]) for i in range(len(pairs))]
    sem_map = {i: s for i, s in pairs}
    return pairs, sem_map

def bm25_scores_all(query, bm25_index, faq_data):
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25_index.get_scores(processed)
    norm_scores = minmax_norm(scores).tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    rank_map = {doc_idx: r+1 for r, doc_idx in enumerate(order)}
    return scores, norm_scores, rank_map

def make_sem_rank_map(sem_pairs, corpus_size):
    order = sorted(range(len(sem_pairs)), key=lambda i: -sem_pairs[i][1])
    rank_map = {}
    for r, pos in enumerate(order):
        doc_idx = sem_pairs[pos][0]
        rank_map[doc_idx] = r+1
    default_rank = corpus_size + TOP_K + 1
    return rank_map, default_rank

def method_confidence_from_margin(scores):
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    if scores is None or len(scores) == 0:
        return 0.0
    if len(scores) == 1:
        return 1.0
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    return max(0.0, min(1.0, margin))

# --------------------- RETRIEVAL METHODS --------------------- #
def bm25_only(query, bm25_index, faq_data):
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25_index.get_scores(processed)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
    return [faq_data[i]["question"] for i in top_indices]

def semantic_only(query, vector_store):
    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    return [doc.metadata["original_question"] for doc, _ in results]

def hybrid_rrf(query, vector_store, bm25_index, faq_data, rrf_k=RRF_K):
    N = len(faq_data)
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    bm25_raw = bm25_index.get_scores(processed)
    bm25_order = sorted(range(N), key=lambda i: -bm25_raw[i])
    bm25_rank = {doc_idx: r+1 for r, doc_idx in enumerate(bm25_order)}
    sem_pairs, _ = semantic_results(query, vector_store, top_k=TOP_K)
    sem_rank, sem_default_rank = make_sem_rank_map(sem_pairs, N)
    fused = []
    for i in range(N):
        r_b = bm25_rank.get(i, N + 1)
        r_s = sem_rank.get(i, sem_default_rank)
        score = 1.0 / (rrf_k + r_b) + 1.0 / (rrf_k + r_s)
        fused.append((score, i))
    top = sorted(fused, key=lambda x: -x[0])[:TOP_K]
    return [faq_data[i]["question"] for _, i in top]

def hybrid_adaptive_global(query, vector_store, bm25_index, faq_data):
    N = len(faq_data)
    bm25_raw, bm25_norm, _ = bm25_scores_all(query, bm25_index, faq_data)
    sem_pairs, sem_map = semantic_results(query, vector_store, top_k=TOP_K)
    sem_norm = np.zeros(N, dtype=float)
    for idx, s in sem_pairs:
        sem_norm[idx] = s
    sem_conf = method_confidence_from_margin(sem_norm)
    bm25_conf = method_confidence_from_margin(bm25_norm)
    sem_w, bm25_w = softmax([sem_conf, bm25_conf], temperature=0.5)
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    combined_scores = []
    for i, item in enumerate(faq_data):
        kb = sum(1 for w in processed if w in item.get("keywords", [])) * 0.05
        score = sem_w * sem_norm[i] + bm25_w * bm25_norm[i] + kb
        combined_scores.append((score, i))
    sorted_final = sorted(combined_scores, key=lambda x: -x[0])[:TOP_K]
    return [faq_data[i]["question"] for _, i in sorted_final]

def hybrid_adaptive_dynamic(query, vector_store, bm25_index, faq_data, top_k=TOP_K):
    # N = len(faq_data)
    # processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # _, bm25_norm, _ = bm25_scores_all(query, bm25_index, faq_data)
    # sem_pairs, _ = semantic_results(query, vector_store, top_k=top_k)
    # sem_norm = np.zeros(N, dtype=float)
    # for idx, s in sem_pairs:
    #     sem_norm[idx] = s
    # bm25_conf = method_confidence_from_margin(bm25_norm)
    # sem_conf = method_confidence_from_margin(sem_norm)
    # alpha = sem_conf / (sem_conf + bm25_conf + 1e-9)
    # beta = bm25_conf / (sem_conf + bm25_conf + 1e-9)
    # combined_scores = []
    # for i, item in enumerate(faq_data):
    #     kb = sum(1 for w in processed if w in item.get("keywords", [])) * 0.05
    #     score = alpha * sem_norm[i] + beta * bm25_norm[i] + kb
    #     combined_scores.append((score, i))
    # top = sorted(combined_scores, key=lambda x: -x[0])[:top_k]
    # return [faq_data[i]["question"] for _, i in top]
    N = len(faq_data)
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()

    # BM25 scores normalized
    _, bm25_norm, _ = bm25_scores_all(query, bm25_index, faq_data)

    # Semantic top-K scores
    sem_pairs, _ = semantic_results(query, vector_store, top_k=top_k)
    sem_norm = np.zeros(N, dtype=float)
    for idx, s in sem_pairs:
        sem_norm[idx] = s

    # Apply gamma scaling for emphasis
    bm25_scaled = np.power(bm25_norm, gamma)
    sem_scaled = np.power(sem_norm, gamma)

    # Softmax for dynamic per-query weighting
    total_bm25 = np.sum(bm25_scaled)
    total_sem = np.sum(sem_scaled)
    alpha = total_sem / (total_sem + total_bm25 + 1e-9)
    beta = total_bm25 / (total_sem + total_bm25 + 1e-9)

    combined_scores = []
    for i, item in enumerate(faq_data):
        kb = sum(1 for w in processed if w in item.get("keywords", [])) * 0.1
        score = alpha * sem_scaled[i] + beta * bm25_scaled[i] + kb
        combined_scores.append((score, i))

    top = sorted(combined_scores, key=lambda x: -x[0])[:top_k]
    return [faq_data[i]["question"] for _, i in top]

# --------------------- EVALUATION --------------------- #
def evaluate(method_fn, test_queries, *args):
    metrics = {'top1_acc': 0, 'recall@3': 0, 'response_times': []}
    for sample in test_queries:
        start_time = time.time()
        retrieved = method_fn(sample["query"], *args)
        metrics['response_times'].append((time.time() - start_time) * 1000)
        expected_list = sample["expected"] if isinstance(sample["expected"], list) else [sample["expected"]]
        if retrieved and retrieved[0] in expected_list:
            metrics['top1_acc'] += 1
        if any(r in retrieved[:3] for r in expected_list):
            metrics['recall@3'] += 1
    total = len(test_queries)
    metrics['top1_acc'] = metrics['top1_acc'] / total * 100
    metrics['recall@3'] = metrics['recall@3'] / total * 100
    metrics['avg_time'] = sum(metrics['response_times']) / total
    return metrics

# --------------------- LOGGING --------------------- #
def log_results_to_json(results, hybrid_details, config, file_path="ll.json"):
    log_entry = {
        "TOP_K": config.get("TOP_K", TOP_K),
        "SEMANTIC_THRESHOLD": config.get("SEMANTIC_THRESHOLD", SEMANTIC_THRESHOLD),
        "RRF_K": config.get("RRF_K", RRF_K),
        "total_queries": hybrid_details["total_queries"],
        "results": results
    }
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(log_entry)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# --------------------- MAIN EXECUTION --------------------- #
if __name__ == "__main__":
    faq_data, test_queries = load_data()
    embeddings = SentenceTransformerEmbeddings()
    vector_store = create_vector_store(embeddings, faq_data)
    bm25_index = create_bm25_index(faq_data)

    results = {}
    # methods = [
    #     ("BM25 Only", bm25_only, [bm25_index, faq_data]),
    #     ("Semantic Only", semantic_only, [vector_store]),
    #     ("Hybrid (RRF)", hybrid_rrf, [vector_store, bm25_index, faq_data]),
    #     ("Hybrid (Adaptive-Global)", hybrid_adaptive_global, [vector_store, bm25_index, faq_data]),
    #     ("Hybrid (Adaptive-Dynamic)", hybrid_adaptive_dynamic, [vector_store, bm25_index, faq_data])
    # ]
    methods = [
        ("BM25 Only", bm25_only, [bm25_index, faq_data]),
        ("Semantic Only", semantic_only, [vector_store]),
        ("Hybrid (RRF)", hybrid_rrf, [vector_store, bm25_index, faq_data]),
        ("Hybrid (Adaptive-Global)", hybrid_adaptive_global, [vector_store, bm25_index, faq_data]),
        ("Hybrid (Adaptive-Dynamic)", hybrid_adaptive_dynamic, [vector_store, bm25_index, faq_data]),  # gamma will use default 2.0
    ]

    for name, fn, args in methods:
        results[name] = evaluate(fn, test_queries, *args)

    print("\nEvaluation Results:")
    print("| Method                   | Top-1 Acc | Recall@3 | Avg Time (ms) |")
    print("|--------------------------|-----------|----------|---------------|")
    for name, metrics in results.items():
        print(f"| {name:<24} | {metrics['top1_acc']:>8.2f}% | {metrics['recall@3']:>8.2f}% | {metrics['avg_time']:>13.2f} |")

    primary_hybrid_key = "Hybrid (Adaptive-Dynamic)"
    hybrid_metrics = results[primary_hybrid_key]
    config = {"TOP_K": TOP_K, "SEMANTIC_THRESHOLD": SEMANTIC_THRESHOLD, "RRF_K": RRF_K}
    results[primary_hybrid_key]["total_queries"] = len(test_queries)
    log_results_to_json(results, results[primary_hybrid_key], config)

    print(f"\n{primary_hybrid_key} Details:")
    print(f"- Top-1 Accuracy: {hybrid_metrics['top1_acc']:.2f}%")
    print(f"- Recall@3: {hybrid_metrics['recall@3']:.2f}%")
    print(f"- Average Response Time: {hybrid_metrics['avg_time']:.2f} ms")
    print(f"- Total Queries Processed: {len(test_queries)}")
