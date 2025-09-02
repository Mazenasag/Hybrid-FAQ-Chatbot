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
BASE_HYBRID_WEIGHTS = (0.7, 0.3)  # Base weights for hybrid approach

# --------------------- CUSTOM EMBEDDINGS CLASS --------------------- #
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# --------------------- QUERY ANALYSIS FUNCTIONS --------------------- #
def analyze_query_type(query):
    """
    Analyze the query to determine its characteristics
    Returns a dictionary with query type scores
    """
    # Convert to lowercase for analysis
    query_lower = query.lower()
    
    # Initialize scores
    scores = {
        'keyword_density': 0,
        'completeness': 0,
        'specificity': 0,
        'ambiguity': 0
    }
    
    # Calculate keyword density (ratio of content words to total words)
    words = query_lower.split()
    content_words = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'why', 'can', 'do', 'is', 'are']]
    scores['keyword_density'] = len(content_words) / max(1, len(words))
    
    # Calculate completeness (query length and structure)
    scores['completeness'] = min(1.0, len(words) / 10)  # Longer queries are more complete
    
    # Calculate specificity (presence of specific terms like numbers, proper nouns)
    specific_indicators = bool(re.search(r'\d+', query)) or bool(re.search(r'[A-Z][a-z]+', query))
    scores['specificity'] = 1.0 if specific_indicators else 0.3
    
    # Calculate ambiguity (presence of vague terms)
    vague_terms = ['something', 'stuff', 'things', 'about', 'maybe', 'possibly', 'perhaps']
    vague_count = sum(1 for term in vague_terms if term in query_lower)
    scores['ambiguity'] = min(1.0, vague_count * 0.3)
    
    return scores

def determine_optimal_weights(query_scores):
    """
    Determine optimal weights based on query characteristics
    Returns (semantic_weight, bm25_weight)
    """
    # Base weights
    semantic_weight, bm25_weight = BASE_HYBRID_WEIGHTS
    
    # Adjust weights based on query characteristics
    if query_scores['keyword_density'] > 0.6:
        # Keyword-rich queries benefit more from BM25
        bm25_weight += 0.2
        semantic_weight -= 0.2
    elif query_scores['ambiguity'] > 0.5:
        # Ambiguous queries benefit more from semantic search
        semantic_weight += 0.2
        bm25_weight -= 0.2
    elif query_scores['completeness'] < 0.4:
        # Incomplete queries benefit more from BM25
        bm25_weight += 0.15
        semantic_weight -= 0.15
    elif query_scores['specificity'] > 0.7:
        # Specific queries benefit from a balanced approach
        # Slight adjustment toward semantic for specific but potentially paraphrased queries
        semantic_weight += 0.1
        bm25_weight -= 0.1
    
    # Ensure weights are within valid range [0, 1] and sum to 1
    semantic_weight = max(0.1, min(0.9, semantic_weight))
    bm25_weight = max(0.1, min(0.9, bm25_weight))
    
    # Normalize to sum to 1
    total = semantic_weight + bm25_weight
    semantic_weight /= total
    bm25_weight /= total
    
    return semantic_weight, bm25_weight

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

# --------------------- RETRIEVAL METHODS --------------------- #
def bm25_only(query, bm25_index, faq_data):
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    scores = bm25_index.get_scores(processed)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
    return [faq_data[i]["question"] for i in top_indices]

def semantic_only(query, vector_store):
    results = vector_store.similarity_search_with_score(query, k=TOP_K)
    return [doc.metadata["original_question"] for doc, _ in results]

def hybrid(query, vector_store, bm25_index, faq_data):
    # Analyze query to determine optimal weights
    query_scores = analyze_query_type(query)
    semantic_weight, bm25_weight = determine_optimal_weights(query_scores)
    
    # Get semantic results
    sem_results = vector_store.similarity_search_with_score(query, k=TOP_K)
    sem_scores = {doc.metadata["original_question"]: 1 - (score/2) for doc, score in sem_results}

    # Get BM25 results
    processed = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    bm25_scores = bm25_index.get_scores(processed)

    # Combine scores with adaptive weights
    combined_scores = []
    for idx, item in enumerate(faq_data):
        sem_score = sem_scores.get(item["question"], 0)
        norm_bm25 = (bm25_scores[idx] - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores) + 1e-9)
        
        # Optional keyword boost if query words are in metadata keywords
        keyword_boost = 0
        if "keywords" in item:
            keyword_boost = sum(1 for w in processed if w in item.get("keywords", [])) * 0.05
        
        # Use adaptive weights
        final_score = semantic_weight * sem_score + bm25_weight * norm_bm25 + keyword_boost
        combined_scores.append((final_score, idx))

    sorted_final = sorted(combined_scores, key=lambda x: -x[0])[:TOP_K]
    return [faq_data[i]["question"] for _, i in sorted_final]

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
        if any(r in retrieved[:2] for r in expected_list):
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
        ("Hybrid (Adaptive)", hybrid, [vector_store, bm25_index, faq_data])
    ]

    for name, fn, args in methods:
        results[name] = evaluate(fn, test_queries, *args)

    # Print results
    print("\nEvaluation Results:")
    print("| Method               | Top-1 Acc | Recall@3 | Avg Time (ms) |")
    print("|----------------------|-----------|----------|---------------|")
    for name, metrics in results.items():
        print(f"| {name:<20} | {metrics['top1_acc']:>8.2f}% | {metrics['recall@3']:>8.2f}% | {metrics['avg_time']:>13.2f} |")

    hybrid_metrics = results["Hybrid (Adaptive)"]
    config = {
        "TOP_K": TOP_K,
        "SEMANTIC_THRESHOLD": SEMANTIC_THRESHOLD,
        "BASE_HYBRID_WEIGHTS": BASE_HYBRID_WEIGHTS
    }
    results["Hybrid (Adaptive)"]["total_queries"] = len(test_queries)
    log_results_to_json(results, results["Hybrid (Adaptive)"], config)

    print("\nHybrid Method Details:")
    print(f"- Top-1 Accuracy: {hybrid_metrics['top1_acc']:.2f}%")
    print(f"- Recall@3: {hybrid_metrics['recall@3']:.2f}%")
    print(f"- Average Response Time: {hybrid_metrics['avg_time']:.2f} ms")
    print(f"- Total Queries Processed: {len(test_queries)}")