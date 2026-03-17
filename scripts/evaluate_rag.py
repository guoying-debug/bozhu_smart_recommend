import os
import sys
import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm
import json

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Set HF Mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from app.core.data_loader import get_data
from app.core.recommender import get_similar_titles, init_recommender
from app.core.llm import analyze_title_with_llm, dashscope
from dashscope import Generation

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_retrieval(sample_size=20):
    """
    Evaluate Retrieval Performance (Recall@K) using pseudo-ground truth.
    Method: Take a known video title, paraphrase it (or use it as is), 
    and check if the original video is retrieved.
    """
    print("\n" + "="*50)
    print(f"🚀 Starting Retrieval Evaluation (Sample Size: {sample_size})")
    print("="*50)
    
    df = get_data()
    if df is None or df.empty:
        print("Error: No data found.")
        return

    # Initialize recommender
    init_recommender()
    
    # Sample data
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    hits_at_5 = 0
    hits_at_10 = 0
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating Retrieval"):
        original_title = row['title']
        
        # Simulate a user query: simply use the title (Ideal case) 
        # In a real scenario, we might paraphrase this.
        # Let's try to remove 20% of characters to simulate a partial query? 
        # Or just use the exact title to test if the vector DB works perfectly.
        # Let's use the exact title for sanity check first.
        query = original_title
        
        # Get recommendations
        try:
            results = get_similar_titles(query, top_k=10)
            retrieved_titles = [r['title'] for r in results]
            
            # Check for hit (exact match)
            # Note: Since we are querying with the exact title, the original video SHOULD be rank 1 (distance 0)
            # unless we excluded it. Our recommender doesn't explicitly exclude the query itself.
            
            if original_title in retrieved_titles[:5]:
                hits_at_5 += 1
            if original_title in retrieved_titles[:10]:
                hits_at_10 += 1
                
        except Exception as e:
            logger.error(f"Error processing {original_title}: {e}")

    recall_5 = hits_at_5 / sample_size
    recall_10 = hits_at_10 / sample_size
    
    print(f"\n📊 Retrieval Metrics:")
    print(f"Recall@5:  {recall_5:.2%}")
    print(f"Recall@10: {recall_10:.2%}")
    print("-" * 50)

def llm_judge_score(user_title, ai_diagnosis, ai_suggestions):
    """
    Use LLM as a Judge to evaluate the quality of the advice.
    Mimics RAGAS 'Answer Relevance' and 'Faithfulness'.
    """
    if not dashscope.api_key:
        return 0, "No API Key"

    prompt = f"""
    You are an expert evaluator for a Video Title Optimization AI.
    
    User Input Title: "{user_title}"
    
    AI Diagnosis: "{ai_diagnosis}"
    AI Suggestions: {ai_suggestions}
    
    Please evaluate the AI's response on a scale of 1 to 5 based on the following criteria:
    1. **Relevance**: Does the diagnosis directly address the specific title provided?
    2. **Actionability**: Are the suggestions concrete and usable?
    3. **Professionalism**: Is the tone professional and helpful?
    
    Output ONLY a JSON object with the following format:
    {{
        "score": <int 1-5>,
        "reason": "<short explanation>"
    }}
    """
    
    try:
        response = Generation.call(
            model=Generation.Models.qwen_plus,
            prompt=prompt,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            try:
                res = json.loads(content)
                return res.get('score', 3), res.get('reason', 'Parsed')
            except:
                return 3, "JSON Parse Error"
        return 0, "API Error"
    except Exception as e:
        return 0, str(e)

def evaluate_generation(sample_size=5):
    """
    Evaluate Generation Quality using LLM-as-a-Judge.
    """
    print("\n" + "="*50)
    print(f"🧠 Starting Generation Evaluation (LLM-as-a-Judge) (Sample Size: {sample_size})")
    print("="*50)
    
    df = get_data()
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=99)
    
    total_score = 0
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating Generation"):
        title = row['title']
        category = row['category'] if pd.notnull(row['category']) else "未知"
        
        # 1. Run the full pipeline (RAG + Prediction + LLM)
        # Mocking prediction inputs for speed
        pred_view = row['view_count'] # Assume prediction is perfect for this test
        feature_explanations = [{"feature": "Test", "effect": "Neutral", "reason": "Testing"}]
        
        # Get Context (RAG)
        similar_titles = get_similar_titles(title, top_k=3)
        
        # Generate Answer
        result = analyze_title_with_llm(
            title=title,
            category=category,
            predicted_view=pred_view,
            feature_explanations=feature_explanations,
            similar_titles=similar_titles
        )
        
        diagnosis = result.get('diagnosis', '')
        suggestions = result.get('suggestions', [])
        
        # 2. Judge
        score, reason = llm_judge_score(title, diagnosis, suggestions)
        total_score += score
        
        print(f"\nTitle: {title[:30]}...")
        print(f"Score: {score}/5 | Reason: {reason}")
        
    avg_score = total_score / sample_size
    print(f"\n🏆 Average RAGAS-style Quality Score: {avg_score:.2f} / 5.0")
    print("="*50)

if __name__ == "__main__":
    evaluate_retrieval(sample_size=20)
    evaluate_generation(sample_size=3)
