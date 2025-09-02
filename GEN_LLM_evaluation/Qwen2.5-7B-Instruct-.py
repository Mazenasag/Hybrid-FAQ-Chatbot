import json
import os
import time
from together import Together
from dotenv import load_dotenv

# --------------------- CONFIGURATION --------------------- #
INPUT_FILE = "qwithout.json"
OUTPUT_FILE = "Qwen2_evaluation.json"
TOGETHER_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

BASE_DELAY = 15  # Base delay between LLM requests to prevent rate limiting

# --------------------- LOAD ENVIRONMENT VARIABLES --------------------- #
load_dotenv()

# --------------------- LLM RESPONSE GENERATION --------------------- #
def generate_llm_response(user_query, static_answer, client, retries=3):
    """
    Generates a conversational response from the static answer
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Rephrase this customer support answer to be more conversational while keeping all facts accurate. Maintain the same meaning but use friendlier language."
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {user_query}\n\nKnowledge base answer: {static_answer}\n\nRephrased answer:"
                    }
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** attempt) * BASE_DELAY
                print(f"Rate limit hit, retrying after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"LLM Error: {str(e)}")
                break
    return static_answer  # Fallback to original answer

# --------------------- MAIN PROCESS --------------------- #
def regenerate_answers():
    # Load FAQ data
    with open(INPUT_FILE) as f:
        faq_data = json.load(f)
    print(f"✅ Loaded {len(faq_data)} FAQ items from {INPUT_FILE}")
    
    # Initialize LLM client
    print("🔄 Initializing LLM client...")
    llm_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    # Process each question-answer pair
    results = []
    for i, item in enumerate(faq_data):
        start_time = time.time()
        question = item["question"]
        static_answer = item["answer"]
        
        print(f"\n🔍 Processing [{i+1}/{len(faq_data)}]: {question[:50]}...")
        print("✨ Generating conversational response...")
        
        # Generate conversational response
        response = generate_llm_response(question, static_answer, llm_client)
        
        # Record results
        elapsed = time.time() - start_time
        results.append({
            "original_question": question,
            "static_answer": static_answer,
            "conversational_answer": response,
            "processing_time": round(elapsed, 2)
        })
        
        # Add delay to avoid rate limiting (skip for last item)
        if i < len(faq_data) - 1:
            print(f"⏳ Adding {BASE_DELAY}s delay...")
            time.sleep(BASE_DELAY)
    
    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Successfully processed {len(results)} questions")
    print(f"💾 Results saved to {OUTPUT_FILE}")
    
    return results

if __name__ == "__main__":
    regenerate_answers()