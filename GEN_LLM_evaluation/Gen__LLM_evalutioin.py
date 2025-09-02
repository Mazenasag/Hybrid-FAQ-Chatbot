import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
from thefuzz import fuzz
import statistics

# Install required packages: pip install nltk numpy thefuzz
# Run: python -c "import nltk; nltk.download('punkt')"

def evaluate_conversational_quality(static, conversational):
    """Evaluate how well static answer is converted to conversational style"""
    if static == conversational:
        return 0
    
    # Calculate similarity
    similarity = fuzz.ratio(static, conversational)
    
    # Length ratio (conversational should be 10-50% longer)
    len_ratio = len(conversational) / max(len(static), 1)
    
    # Detect conversational markers
    markers = ["hi", "hello", "hey", "thanks", "please", "welcome", 
              "happy", "glad", "sure", "absolutely", "definitely",
              "don't worry", "no problem", "feel free", "let's", "we'd"]
    markers_count = sum(1 for marker in markers if marker in conversational.lower())
    
    # Calculate component scores
    similarity_score = max(0, 100 - similarity) / 100  # Lower similarity is better
    length_score = min(1, max(0, (len_ratio - 1) * 2))  # Reward 10-50% longer responses
    markers_score = min(1, markers_count / 4)  # Reward conversational markers
    
    return 0.5 * similarity_score + 0.3 * length_score + 0.2 * markers_score

def calculate_bleu(reference, candidate):
    """Calculate BLEU score against ChatGPT reference"""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, 
                         smoothing_function=smoothie,
                         weights=(0.5, 0.3, 0.2, 0))  # Weighted n-grams

def load_data(file_path):
    """Load JSON data and index by question"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {item['original_question']: item for item in data}

def evaluate_model(model_data, chatgpt_data):
    """Evaluate a single model against ChatGPT reference"""
    bleu_scores = []
    quality_scores = []
    processing_times = []
    exact_matches = 0
    valid_questions = 0
    
    for question, item in model_data.items():
        if question not in chatgpt_data:
            continue
            
        chatgpt_ref = chatgpt_data[question]['conversational_answer']
        static = item['static_answer']
        conversational = item['conversational_answer']
        processing_time = item.get('processing_time', 0)
        
        # Skip empty responses
        if not conversational.strip():
            continue
            
        # Calculate metrics
        bleu_score = calculate_bleu(chatgpt_ref, conversational)
        quality_score = evaluate_conversational_quality(static, conversational)
        
        bleu_scores.append(bleu_score)
        quality_scores.append(quality_score)
        processing_times.append(processing_time)
        
        if static == conversational:
            exact_matches += 1
            
        valid_questions += 1
    
    # Handle case with no valid questions
    if valid_questions == 0:
        return {
            'avg_bleu': 0,
            'avg_quality': 0,
            'avg_processing_time': 0,
            'exact_match_pct': 0,
            'num_questions': 0
        }
    
    return {
        'avg_bleu': np.mean(bleu_scores),
        'bleu_stdev': np.std(bleu_scores),
        'avg_quality': np.mean(quality_scores),
        'quality_stdev': np.std(quality_scores),
        'avg_processing_time': np.mean(processing_times),
        'processing_stdev': np.std(processing_times),
        'exact_match_pct': (exact_matches / valid_questions) * 100,
        'num_questions': valid_questions
    }

def normalize_scores(results):
    """Normalize metrics for composite score calculation"""
    metrics = ['avg_bleu', 'avg_quality', 'avg_processing_time']
    ranges = {metric: {'min': float('inf'), 'max': float('-inf')} for metric in metrics}
    
    # Find min/max ranges across models
    for model in results.values():
        for metric in metrics:
            if metric in model:
                ranges[metric]['min'] = min(ranges[metric]['min'], model[metric])
                ranges[metric]['max'] = max(ranges[metric]['max'], model[metric])
    
    # Apply normalization
    for model in results:
        data = results[model]
        
        # Normalize BLEU and Quality (higher is better)
        data['norm_bleu'] = (data['avg_bleu'] - ranges['avg_bleu']['min']) / max(
            (ranges['avg_bleu']['max'] - ranges['avg_bleu']['min']), 1e-9)
        
        data['norm_quality'] = (data['avg_quality'] - ranges['avg_quality']['min']) / max(
            (ranges['avg_quality']['max'] - ranges['avg_quality']['min']), 1e-9)
        
        # Normalize processing time (lower is better)
        time_range = ranges['avg_processing_time']['max'] - ranges['avg_processing_time']['min']
        if time_range > 0:
            data['norm_time'] = 1 - ((data['avg_processing_time'] - ranges['avg_processing_time']['min']) / time_range)
        else:
            data['norm_time'] = 1
        
        # Calculate composite score
        data['composite_score'] = (
            0.45 * data['norm_bleu'] + 
            0.45 * data['norm_quality'] + 
            0.10 * data['norm_time']
        )
    
    return results

def main():
    # Configuration - update with your file paths
    model_files = {
        'meta-llama': 'GEN_LLM_evaluation\meta-llama_evaluation.json',
        'mistralai': 'GEN_LLM_evaluation\mistralai_evaluation.json',
        'Qwen2': 'GEN_LLM_evaluation\Qwen2_evaluation.json'
    }
    chatgpt_file = 'GEN_LLM_evaluation\qwith_conversational.json'
    
    # Load ChatGPT gold standard
    try:
        chatgpt_data = load_data(chatgpt_file)
    except FileNotFoundError:
        print(f"Error: ChatGPT reference file {chatgpt_file} not found!")
        return
    
    # Evaluate all models
    results = {}
    for model_name, file_path in model_files.items():
        try:
            model_data = load_data(file_path)
            results[model_name] = evaluate_model(model_data, chatgpt_data)
        except FileNotFoundError:
            print(f"Error: Model file {file_path} not found! Skipping {model_name}")
            results[model_name] = {}
    
    # Normalize scores and calculate composite
    results = normalize_scores(results)
    
    # Add ranking
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1].get('composite_score', 0), 
                          reverse=True)
    results['ranking'] = [model[0] for model in sorted_models]
    
    # Save results
    with open('Gen_llm_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation complete! Results saved to Gen_llm_evaluation.json")
    
    # Print summary
    print("\nModel Evaluation Summary:")
    print(f"{'Model':<12} {'Comp Score':<10} {'BLEU':<8} {'Quality':<8} {'Time(ms)':<10} {'Exact Match%'}")
    print("-" * 60)
    for model in sorted_models:
        data = model[1]
        print(f"{model[0]:<12} {data.get('composite_score', 0):.4f}    "
              f"{data.get('avg_bleu', 0):.4f}   {data.get('avg_quality', 0):.4f}  "
              f"{data.get('avg_processing_time', 0):>8.2f}     "
              f"{data.get('exact_match_pct', 0):>5.1f}%")

if __name__ == "__main__":
    main()