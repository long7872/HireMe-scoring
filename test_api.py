import requests
import json

API_URL = "http://localhost:8000"

test_transcript = """
Question: How would you design a distributed cache system?

Response: So, um, for a distributed cache, I think the key challenge is maintaining consistency across multiple nodes while keeping it fast, right? 

I would start with Redis as the caching layer because it's in-memory and super fast. For distribution, I'd probably use consistent hashing to distribute keys across nodes - this way when we add or remove nodes, we minimize the number of keys that need to be moved.

For the architecture, we'd have multiple cache nodes behind a load balancer. The client would hash the key and the load balancer would route to the appropriate node. We'd also need a strategy for cache invalidation - maybe TTL-based or event-driven using something like Kafka.

One important thing is handling cache misses. When a key isn't in the cache, we need to fetch from the database and populate the cache. We should use cache-aside pattern to avoid thundering herd problem.

For high availability, we could use Redis Sentinel for automatic failover. And for persistence, we might want to use Redis AOF to prevent data loss during restarts.
"""

def test_combined():
    """Test ensemble assessment"""
    print("="*80)
    print("TESTING ENSEMBLE (BERT + Gemini)")
    print("="*80 + "\n")
    
    payload = {
        "text": test_transcript,
        "include_confidence": True,
        "use_ensemble": True
    }
    
    response = requests.post(f"{API_URL}/assess", json=payload)
    result = response.json()
    
    print(f"Status: {response.status_code}")
    print(f"\n📊 RESULTS:")
    print(f"{'='*80}")
    print(f"Method Used: {result['method_used']}")
    print(f"Overall Score: {result['overall_score']}/5")
    print(f"  - BERT Overall: {result['bert_overall']}/5")
    if result['gemini_overall']:
        print(f"  - Gemini Overall: {result['gemini_overall']}/5")
    print(f"\nRecommendation: {result['recommendation']}")
    
    print(f"\n📋 TRAIT SCORES:")
    print(f"{'='*80}")
    for trait in result['trait_scores']:
        print(f"\n{trait['trait']} (Priority #{trait['priority']}):")
        print(f"  BERT:     {trait['bert_score']}/5", end="")
        if trait['confidence']:
            print(f" (confidence: {trait['confidence']:.2%})")
        else:
            print()
        
        if trait['gemini_score']:
            print(f"  Gemini:   {trait['gemini_score']}/5")
        
        print(f"  Ensemble: {trait['ensemble_score']}/5 ⭐")

def test_bert_only():
    """Test BERT-only assessment"""
    print("\n" + "="*80)
    print("TESTING BERT ONLY (Faster)")
    print("="*80 + "\n")
    
    payload = {
        "text": test_transcript,
        "include_confidence": False
    }
    
    response = requests.post(f"{API_URL}/assess-bert-only", json=payload)
    result = response.json()
    
    print(f"Overall Score: {result['overall_score']}/5")
    print(f"Method: {result['method_used']}")
    print(f"Recommendation: {result['recommendation']}")

if __name__ == "__main__":
    test_combined()
    test_bert_only()