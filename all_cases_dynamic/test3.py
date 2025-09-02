import string
import time
import os
import json
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.embeddings.base import Embeddings
from scipy import stats

# --------------------- CONFIGURATION --------------------- #
TOP_K = 20
SEMANTIC_THRESHOLD = 0.6
BASE_HYBRID_WEIGHTS = (0.7, 0.3)  # Base weights for fallback
RRF_K = 60  # Constant for RRF

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
    test_path = r"all_cases/test.json"

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

# --------------------- HELPER FUNCTIONS --------------------- #
def semantic_results(query, vector_store, top_k=TOP_K):
    """Get semantic search results with indices"""
    results = vector_store.similarity_search_with_score(query, k=top_k)
    pairs = [(doc.metadata["index"], score) for doc, score in results]
    return pairs, len(faq_data)

def make_sem_rank_map(sem_pairs, N):
    """Create a rank mapping for semantic results"""
    sem_rank = {}
    for rank, (doc_idx, _) in enumerate(sem_pairs):
        sem_rank[doc_idx] = rank + 1
    sem_default_rank = N + 1  # Default rank for documents not in top-K
    return sem_rank, sem_default_rank

# --------------------- RETRIEVAL METHODS --------------------- #
def bm25_only(query, bm25_index, faq_data):
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25_index.get_scores(processed)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
    return [faq_data[i]["question"] for i in top_indices]

def semantic_only(query, vector_store):
    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    return [doc.metadata["original_question"] for doc, _ in results]

def analyze_query_characteristics(query, bm25_scores, sem_scores):
    """
    Analyze query characteristics to determine optimal weighting strategy
    """
    # Convert semantic distance to similarity (higher is better)
    sem_similarities = [1 - (score / 2) for score in sem_scores]
    
    # Calculate basic statistics
    bm25_mean = np.mean(bm25_scores)
    bm25_std = np.std(bm25_scores)
    bm25_max = np.max(bm25_scores)
    
    sem_mean = np.mean(sem_similarities)
    sem_std = np.std(sem_similarities)
    sem_max = np.max(sem_similarities)
    
    # Calculate separation between top results
    bm25_top_sep = (bm25_max - np.partition(bm25_scores, -2)[-2]) / max(bm25_max, 1e-9)
    sem_top_sep = (sem_max - np.partition(sem_similarities, -2)[-2]) / max(sem_max, 1e-9)
    
    # Analyze query length and structure
    query_words = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    query_length = len(query_words)
    
    # Check for specific patterns that favor BM25
    has_specific_terms = any(len(word) > 5 for word in query_words) or any(word.isdigit() for word in query_words)
    has_common_terms = any(word in ['what', 'how', 'when', 'where', 'why', 'can', 'do', 'is', 'are'] for word in query_words)
    
    # Determine optimal strategy
    characteristics = {
        'bm25_confidence': min(1.0, bm25_top_sep * 2 + (1 - bm25_std/bm25_mean) if bm25_mean > 0 else 0.5),
        'semantic_confidence': min(1.0, sem_top_sep * 2 + (1 - sem_std/sem_mean) if sem_mean > 0 else 0.5),
        'query_specificity': 1.0 if has_specific_terms else 0.3,
        'query_completeness': min(1.0, query_length / 5),
        'favors_bm25': has_specific_terms and not has_common_terms
    }
    
    return characteristics

def calculate_optimal_weights(query_characteristics):
    """
    Calculate optimal weights based on query characteristics
    """
    # Base weights
    sem_weight, bm25_weight = BASE_HYBRID_WEIGHTS
    
    # Adjust based on confidence scores
    sem_weight *= query_characteristics['semantic_confidence']
    bm25_weight *= query_characteristics['bm25_confidence']
    
    # Additional adjustments based on query characteristics
    if query_characteristics['favors_bm25']:
        bm25_weight *= 1.5
    elif query_characteristics['query_completeness'] < 0.6:
        # Incomplete queries often benefit from BM25
        bm25_weight *= 1.3
    else:
        # Complete queries often benefit from semantic search
        sem_weight *= 1.2
    
    # Ensure minimum weights
    sem_weight = max(0.2, min(0.8, sem_weight))
    bm25_weight = max(0.2, min(0.8, bm25_weight))
    
    # Normalize to sum to 1
    total = sem_weight + bm25_weight
    sem_weight /= total
    bm25_weight /= total
    
    return sem_weight, bm25_weight

def hybrid(query, vector_store, bm25_index, faq_data):
    # Get semantic results
    sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
    sem_scores_list = [score for _, score in sem_results]
    sem_scores = {doc.metadata["original_question"]: 1 - (score/2) for doc, score in sem_results}

    # Get BM25 results
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    bm25_scores = bm25_index.get_scores(processed)
    
    # Analyze query characteristics
    query_chars = analyze_query_characteristics(query, bm25_scores, sem_scores_list)
    
    # Calculate optimal weights for this query
    sem_weight, bm25_weight = calculate_optimal_weights(query_chars)
    
    # Combine scores with optimized weights
    combined_scores = []
    for idx, item in enumerate(faq_data):
        sem_score = sem_scores.get(item["question"], 0)
        norm_bm25 = (bm25_scores[idx] - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-9)
        
        # Optional keyword boost if query words are in metadata keywords
        keyword_boost = 0
        if "keywords" in item:
            keyword_boost = sum(1 for w in processed if w in item.get("keywords", [])) * 0.05
        
        # Use optimized weights
        final_score = sem_weight * sem_score + bm25_weight * norm_bm25 + keyword_boost
        combined_scores.append((final_score, idx))

    sorted_final = sorted(combined_scores, key=lambda x: -x[0])[:TOP_K]
    return [faq_data[i]["question"] for _, i in sorted_final]

def hybrid_rrf(query, vector_store, bm25_index, faq_data):
    """
    Reciprocal Rank Fusion:
      score(d) = sum_m 1 / (rrf_k + rank_m(d))
    Robust to scale differences, strong Recall@K in practice.
    """
    N = len(faq_data)

    # BM25 ranks over full corpus
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    bm25_raw = bm25_index.get_scores(processed)
    bm25_order = sorted(range(N), key=lambda i: -bm25_raw[i])
    bm25_rank = {i: r+1 for r, i in enumerate(bm25_order)}

    # Semantic ranks over top-K (others get large rank)
    sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
    sem_pairs = [(doc.metadata["index"], score) for doc, score in sem_results]
    
    sem_rank = {}
    for rank, (doc_idx, _) in enumerate(sem_pairs):
        sem_rank[doc_idx] = rank + 1
    sem_default_rank = N + 1  # Default rank for documents not in top-K

    # RRF fusion
    fused = []
    for i in range(N):
        r_b = bm25_rank.get(i, N + 1)
        r_s = sem_rank.get(i, sem_default_rank)
        score = 1.0 / (RRF_K + r_b) + 1.0 / (RRF_K + r_s)
        fused.append((score, i))

    top = sorted(fused, key=lambda x: -x[0])[:TOP_K]
    return [faq_data[i]["question"] for _, i in top]

# --------------------- EVALUATION --------------------- #
def evaluate(method_fn, test_queries, *args):
    metrics = {'top1_acc': 0, 'recall@3': 0, 'response_times': []}
    for sample in test_queries:
        start_time = time.time()
        retrieved = method_fn(sample["query"], *args)
        metrics['response_times'].append((time.time() - start_time) * 1000)

        expected_list = sample["expected"] if isinstance(sample["expected"], list) else [sample["expected"]]
        if retrieved[0] in expected_list:
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
        "TOP_K": config["TOP_K"],
        "SEMANTIC_THRESHOLD": config["SEMANTIC_THRESHOLD"],
        "BASE_HYBRID_WEIGHTS": config["BASE_HYBRID_WEIGHTS"],
        "RRF_K": config["RRF_K"],
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
    methods = [
        ("BM25 Only", bm25_only, [bm25_index, faq_data]),
        ("Semantic Only", semantic_only, [vector_store]),
        ("Hybrid (Optimized)", hybrid, [vector_store, bm25_index, faq_data]),
        ("Hybrid RRF", hybrid_rrf, [vector_store, bm25_index, faq_data])
    ]

    for name, fn, args in methods:
        results[name] = evaluate(fn, test_queries, *args)

    # Print results
    print("\nEvaluation Results:")
    print("| Method               | Top-1 Acc | Recall@3 | Avg Time (ms) |")
    print("|----------------------|-----------|----------|---------------|")
    for name, metrics in results.items():
        print(f"| {name:<20} | {metrics['top1_acc']:>8.2f}% | {metrics['recall@3']:>8.2f}% | {metrics['avg_time']:>13.2f} |")

    hybrid_metrics = results["Hybrid (Optimized)"]
    config = {
        "TOP_K": TOP_K,
        "SEMANTIC_THRESHOLD": SEMANTIC_THRESHOLD,
        "BASE_HYBRID_WEIGHTS": BASE_HYBRID_WEIGHTS,
        "RRF_K": RRF_K
    }
    results["Hybrid (Optimized)"]["total_queries"] = len(test_queries)
    log_results_to_json(results, results["Hybrid (Optimized)"], config)

    print("\nHybrid Method Details:")
    print(f"- Top-1 Accuracy: {hybrid_metrics['top1_acc']:.2f}%")
    print(f"- Recall@3: {hybrid_metrics['recall@3']:.2f}%")
    print(f"- Average Response Time: {hybrid_metrics['avg_time']:.2f} ms")
    print(f"- Total Queries Processed: {len(test_queries)}")
# import string
# import time
# import os
# import json
# import numpy as np
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from langchain.embeddings.base import Embeddings
# from scipy import stats

# # --------------------- CONFIGURATION --------------------- #
# TOP_K = 20
# SEMANTIC_THRESHOLD = 0.6
# BASE_HYBRID_WEIGHTS = (0.7, 0.3)  # Base weights for fallback
# RRF_K = 60  # Constant for RRF

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
#     test_path = r"all_cases/test.json"

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

# # --------------------- RETRIEVAL METHODS --------------------- #
# def bm25_only(query, bm25_index, faq_data):
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     scores = bm25_index.get_scores(processed)
#     top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
#     return [faq_data[i]["question"] for i in top_indices]

# def semantic_only(query, vector_store):
#     results = vector_store.similarity_search_with_score(query, k=TOP_K)
#     return [doc.metadata["original_question"] for doc, _ in results]

# def analyze_query_characteristics(query, bm25_scores, sem_scores):
#     """
#     Analyze query characteristics to determine optimal weighting strategy
#     """
#     # Convert semantic distance to similarity (higher is better)
#     sem_similarities = [1 - (score / 2) for score in sem_scores]
    
#     # Calculate basic statistics
#     bm25_mean = np.mean(bm25_scores)
#     bm25_std = np.std(bm25_scores)
#     bm25_max = np.max(bm25_scores)
    
#     sem_mean = np.mean(sem_similarities)
#     sem_std = np.std(sem_similarities)
#     sem_max = np.max(sem_similarities)
    
#     # Calculate separation between top results
#     bm25_top_sep = (bm25_max - np.partition(bm25_scores, -2)[-2]) / max(bm25_max, 1e-9)
#     sem_top_sep = (sem_max - np.partition(sem_similarities, -2)[-2]) / max(sem_max, 1e-9)
    
#     # Analyze query length and structure
#     query_words = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     query_length = len(query_words)
    
#     # Check for specific patterns that favor BM25
#     has_specific_terms = any(len(word) > 5 for word in query_words) or any(word.isdigit() for word in query_words)
#     has_common_terms = any(word in ['what', 'how', 'when', 'where', 'why', 'can', 'do', 'is', 'are'] for word in query_words)
    
#     # Determine optimal strategy
#     characteristics = {
#         'bm25_confidence': min(1.0, bm25_top_sep * 2 + (1 - bm25_std/bm25_mean) if bm25_mean > 0 else 0.5),
#         'semantic_confidence': min(1.0, sem_top_sep * 2 + (1 - sem_std/sem_mean) if sem_mean > 0 else 0.5),
#         'query_specificity': 1.0 if has_specific_terms else 0.3,
#         'query_completeness': min(1.0, query_length / 5),
#         'favors_bm25': has_specific_terms and not has_common_terms
#     }
    
#     return characteristics

# def calculate_optimal_weights(query_characteristics):
#     """
#     Calculate optimal weights based on query characteristics
#     """
#     # Base weights
#     sem_weight, bm25_weight = BASE_HYBRID_WEIGHTS
    
#     # Adjust based on confidence scores
#     sem_weight *= query_characteristics['semantic_confidence']
#     bm25_weight *= query_characteristics['bm25_confidence']
    
#     # Additional adjustments based on query characteristics
#     if query_characteristics['favors_bm25']:
#         bm25_weight *= 1.5
#     elif query_characteristics['query_completeness'] < 0.6:
#         # Incomplete queries often benefit from BM25
#         bm25_weight *= 1.3
#     else:
#         # Complete queries often benefit from semantic search
#         sem_weight *= 1.2
    
#     # Ensure minimum weights
#     sem_weight = max(0.2, min(0.8, sem_weight))
#     bm25_weight = max(0.2, min(0.8, bm25_weight))
    
#     # Normalize to sum to 1
#     total = sem_weight + bm25_weight
#     sem_weight /= total
#     bm25_weight /= total
    
#     return sem_weight, bm25_weight

# def hybrid_rrf(query, vector_store, bm25_index, faq_data):
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
#     bm25_rank = {i: r+1 for r, i in enumerate(bm25_order)}

#     # Semantic ranks over top-K (others get large rank)
#     sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
#     sem_pairs = [(doc.metadata["index"], score) for doc, score in sem_results]
    
#     sem_rank = {}
#     for rank, (doc_idx, _) in enumerate(sem_pairs):
#         sem_rank[doc_idx] = rank + 1
#     sem_default_rank = N + 1  # Default rank for documents not in top-K

#     # RRF fusion
#     fused = []
#     for i in range(N):
#         r_b = bm25_rank.get(i, N + 1)
#         r_s = sem_rank.get(i, sem_default_rank)
#         score = 1.0 / (RRF_K + r_b) + 1.0 / (RRF_K + r_s)
#         fused.append((score, i))

#     top = sorted(fused, key=lambda x: -x[0])[:TOP_K]
#     return [faq_data[i]["question"] for _, i in top]

# def hybrid_weighted(query, vector_store, bm25_index, faq_data):
#     """
#     The original weighted hybrid approach
#     """
#     # Get semantic results
#     sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
#     sem_scores_list = [score for _, score in sem_results]
#     sem_scores = {doc.metadata["original_question"]: 1 - (score/2) for doc, score in sem_results}

#     # Get BM25 results
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     bm25_scores = bm25_index.get_scores(processed)
    
#     # Analyze query characteristics
#     query_chars = analyze_query_characteristics(query, bm25_scores, sem_scores_list)
    
#     # Calculate optimal weights for this query
#     sem_weight, bm25_weight = calculate_optimal_weights(query_chars)
    
#     # Combine scores with optimized weights
#     combined_scores = []
#     for idx, item in enumerate(faq_data):
#         sem_score = sem_scores.get(item["question"], 0)
#         norm_bm25 = (bm25_scores[idx] - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-9)
        
#         # Optional keyword boost if query words are in metadata keywords
#         keyword_boost = 0
#         if "keywords" in item:
#             keyword_boost = sum(1 for w in processed if w in item.get("keywords", [])) * 0.05
        
#         # Use optimized weights
#         final_score = sem_weight * sem_score + bm25_weight * norm_bm25 + keyword_boost
#         combined_scores.append((final_score, idx))

#     sorted_final = sorted(combined_scores, key=lambda x: -x[0])[:TOP_K]
#     return [faq_data[i]["question"] for _, i in sorted_final]

# def unified_hybrid(query, vector_store, bm25_index, faq_data):
#     """
#     Unified hybrid approach that combines the strengths of both methods:
#     1. Uses RRF for better top-1 accuracy
#     2. Uses weighted hybrid for better recall@3
#     3. Dynamically selects the best approach based on query characteristics
#     """
#     # Get both sets of results
#     rrf_results = hybrid_rrf(query, vector_store, bm25_index, faq_data)
#     weighted_results = hybrid_weighted(query, vector_store, bm25_index, faq_data)
    
#     # Get semantic results for analysis
#     sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
#     sem_scores_list = [score for _, score in sem_results]
    
#     # Get BM25 scores for analysis
#     processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
#     bm25_scores = bm25_index.get_scores(processed)
    
#     # Analyze query characteristics
#     query_chars = analyze_query_characteristics(query, bm25_scores, sem_scores_list)
    
#     # Determine which method to prioritize based on query characteristics
#     if query_chars['favors_bm25'] or query_chars['query_specificity'] > 0.7:
#         # Favor RRF for specific queries (better top-1)
#         primary_results = rrf_results
#         secondary_results = weighted_results
#         primary_weight = 0.7
#         secondary_weight = 0.3
#     else:
#         # Favor weighted hybrid for general queries (better recall)
#         primary_results = weighted_results
#         secondary_results = rrf_results
#         primary_weight = 0.7
#         secondary_weight = 0.3
    
#     # Create a combined ranking using a weighted Borda count method
#     combined_scores = {}
    
#     # Score based on primary method
#     for rank, question in enumerate(primary_results):
#         idx = next(i for i, item in enumerate(faq_data) if item["question"] == question)
#         combined_scores[idx] = (TOP_K - rank) * primary_weight
    
#     # Score based on secondary method
#     for rank, question in enumerate(secondary_results):
#         idx = next(i for i, item in enumerate(faq_data) if item["question"] == question)
#         if idx in combined_scores:
#             combined_scores[idx] += (TOP_K - rank) * secondary_weight
#         else:
#             combined_scores[idx] = (TOP_K - rank) * secondary_weight
    
#     # Sort by combined score
#     sorted_indices = sorted(combined_scores.keys(), key=lambda x: -combined_scores[x])[:TOP_K]
#     return [faq_data[i]["question"] for i in sorted_indices]

# # --------------------- EVALUATION --------------------- #
# def evaluate(method_fn, test_queries, *args):
#     metrics = {'top1_acc': 0, 'recall@3': 0, 'response_times': []}
#     for sample in test_queries:
#         start_time = time.time()
#         retrieved = method_fn(sample["query"], *args)
#         metrics['response_times'].append((time.time() - start_time) * 1000)

#         expected_list = sample["expected"] if isinstance(sample["expected"], list) else [sample["expected"]]
#         if retrieved[0] in expected_list:
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
#         "TOP_K": config["TOP_K"],
#         "SEMANTIC_THRESHOLD": config["SEMANTIC_THRESHOLD"],
#         "BASE_HYBRID_WEIGHTS": config["BASE_HYBRID_WEIGHTS"],
#         "RRF_K": config["RRF_K"],
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
#         ("Hybrid Weighted", hybrid_weighted, [vector_store, bm25_index, faq_data]),
#         ("Hybrid RRF", hybrid_rrf, [vector_store, bm25_index, faq_data]),
#         ("Unified Hybrid", unified_hybrid, [vector_store, bm25_index, faq_data])
#     ]

#     for name, fn, args in methods:
#         results[name] = evaluate(fn, test_queries, *args)

#     # Print results
#     print("\nEvaluation Results:")
#     print("| Method               | Top-1 Acc | Recall@3 | Avg Time (ms) |")
#     print("|----------------------|-----------|----------|---------------|")
#     for name, metrics in results.items():
#         print(f"| {name:<20} | {metrics['top1_acc']:>8.2f}% | {metrics['recall@3']:>8.2f}% | {metrics['avg_time']:>13.2f} |")

#     unified_metrics = results["Unified Hybrid"]
#     config = {
#         "TOP_K": TOP_K,
#         "SEMANTIC_THRESHOLD": SEMANTIC_THRESHOLD,
#         "BASE_HYBRID_WEIGHTS": BASE_HYBRID_WEIGHTS,
#         "RRF_K": RRF_K
#     }
#     results["Unified Hybrid"]["total_queries"] = len(test_queries)
#     log_results_to_json(results, results["Unified Hybrid"], config)

#     print("\nUnified Hybrid Method Details:")
#     print(f"- Top-1 Accuracy: {unified_metrics['top1_acc']:.2f}%")
#     print(f"- Recall@3: {unified_metrics['recall@3']:.2f}%")
#     print(f"- Average Response Time: {unified_metrics['avg_time']:.2f} ms")
#     print(f"- Total Queries Processed: {len(test_queries)}")