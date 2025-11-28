# ============================================================================
# AI4LIFE COMBINED API - BERT Models + Gemini Ensemble
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from typing import Dict, List, Optional
import logging
import time
import json
import re
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "./models"
MAX_LENGTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIT_NAMES = {
    'problem_solving': 'Problem-Solving',
    'technical_knowledge': 'Technical Knowledge',
    'emotional_intelligence': 'Emotional Intelligence',
    'communication': 'Communication'
}

TRAIT_PRIORITIES = {
    'problem_solving': 1,
    'technical_knowledge': 2,
    'emotional_intelligence': 3,
    'communication': 4
}

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found - Gemini scoring disabled")
    gemini_client = None
else:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API initialized")

# ============================================================================
# GEMINI HELPER FUNCTIONS
# ============================================================================

def clean_json(text):
    """Clean Gemini output and return Python dict"""
    if isinstance(text, dict):
        return text

    text = text.strip()

    # Remove code fences
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Extract JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    # Parse JSON
    try:
        return json.loads(text)
    except Exception as e:
        logger.error(f"❌ JSON parse error: {e}")
        logger.error(f"Raw text: {text}")
        return None

def call_gemini_with_retry(prompt, retries=3, delay=1):
    """Call Gemini with exponential backoff retry"""
    for i in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt
            )
            return response
        except Exception as e:
            logger.warning(f"⚠️ Gemini error (attempt {i+1}/{retries}): {e}")
            if i == retries - 1:
                raise
            sleep_time = delay * (2 ** i)
            logger.info(f"⏳ Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

def build_gemini_prompt(text: str) -> str:
    """Build prompt for Gemini evaluation"""
    return f"""You are an expert IT interview evaluator. Analyze the following interview transcript and provide scores.

IMPORTANT: Return ONLY valid JSON. No explanations, no code fences, no extra text.

Evaluate these 4 traits (score 1-5 for each):

1. **Problem-Solving** (1-5): Ability to break down problems, think analytically, propose solutions
2. **Technical Knowledge** (1-5): Understanding of technical concepts, tools, technologies
3. **Emotional Intelligence** (1-5): Self-awareness, empathy, professionalism in communication
4. **Communication** (1-5): Clarity, structure, ability to explain complex topics

Scoring guide:
- 5: Excellent - Expert level
- 4: Good - Strong understanding
- 3: Average - Satisfactory
- 2: Below Average - Needs improvement
- 1: Poor - Significant gaps

Return ONLY this JSON structure (exact key names):
{{
  "Problem-Solving": 1-5,
  "Technical Knowledge": 1-5,
  "Emotional Intelligence": 1-5,
  "Communication": 1-5
}}

Interview transcript:
{text}

JSON output:"""

def get_gemini_scores(text: str) -> Optional[Dict]:
    """Get trait scores from Gemini"""
    if gemini_client is None:
        return None
    
    try:
        prompt = build_gemini_prompt(text)
        response = call_gemini_with_retry(prompt)
        result = clean_json(response.text)
        
        if result:
            logger.info("✅ Gemini scores obtained")
            return result
        else:
            logger.warning("⚠️ Gemini returned invalid JSON")
            return None
            
    except Exception as e:
        logger.error(f"❌ Gemini scoring failed: {e}")
        return None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AssessmentRequest(BaseModel):
    text: str = Field(..., min_length=10)
    include_confidence: bool = Field(default=False)
    use_ensemble: bool = Field(default=True, description="Use BERT + Gemini ensemble")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "How would you design a caching system? I would use Redis...",
                "include_confidence": True,
                "use_ensemble": True
            }
        }

class TraitScore(BaseModel):
    trait: str
    bert_score: int = Field(..., ge=1, le=5)
    gemini_score: Optional[int] = Field(None, ge=1, le=5)
    ensemble_score: int = Field(..., ge=1, le=5)
    priority: int
    confidence: Optional[float] = None

class AssessmentResponse(BaseModel):
    overall_score: float
    bert_overall: float
    gemini_overall: Optional[float] = None
    trait_scores: List[TraitScore]
    recommendation: str
    method_used: str
    text_length: int

# ============================================================================
# BERT MODEL PREDICTOR
# ============================================================================

class BERTPredictor:
    """BERT-based multi-trait predictor"""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all 4 BERT models"""
        for trait in TRAIT_NAMES.keys():
            try:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=5
                )
                
                model_path = f"{self.model_dir}/balanced_model_{trait}.pth"
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                
                model.to(self.device)
                model.eval()
                
                self.models[trait] = model
                logger.info(f"✅ Loaded BERT {TRAIT_NAMES[trait]} model")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {trait} model: {e}")
                raise
    
    def predict(self, text: str, include_confidence: bool = False) -> Dict:
        """Predict all trait scores using BERT models"""
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        predictions = {}
        
        with torch.no_grad():
            for trait, model in self.models.items():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                predicted_class = torch.argmax(probabilities, dim=1).item()
                score = predicted_class + 1  # Convert 0-4 to 1-5
                confidence = probabilities[0][predicted_class].item()
                
                predictions[trait] = {
                    'score': score,
                    'confidence': confidence if include_confidence else None
                }
        
        return predictions

# ============================================================================
# ENSEMBLE PREDICTOR
# ============================================================================

class EnsemblePredictor:
    """Combine BERT and Gemini predictions"""
    
    def __init__(self, bert_predictor: BERTPredictor):
        self.bert = bert_predictor
    
    def predict(self, text: str, include_confidence: bool = False, 
                use_ensemble: bool = True) -> Dict:
        """
        Get ensemble predictions from BERT + Gemini
        
        Args:
            text: Interview transcript
            include_confidence: Include confidence scores
            use_ensemble: Use ensemble (True) or BERT only (False)
        
        Returns:
            Dictionary with BERT, Gemini, and ensemble scores
        """
        # Get BERT predictions
        bert_preds = self.bert.predict(text, include_confidence)
        
        # Get Gemini predictions (if enabled)
        gemini_preds = None
        if use_ensemble and gemini_client is not None:
            gemini_raw = get_gemini_scores(text)
            if gemini_raw:
                # Map Gemini output to internal format
                gemini_preds = {}
                for trait in TRAIT_NAMES.keys():
                    gemini_key = TRAIT_NAMES[trait]
                    if gemini_key in gemini_raw:
                        gemini_preds[trait] = {
                            'score': int(gemini_raw[gemini_key]),
                            'confidence': None
                        }
        
        # Combine predictions
        ensemble_preds = {}
        
        for trait in TRAIT_NAMES.keys():
            bert_score = bert_preds[trait]['score']
            gemini_score = gemini_preds[trait]['score'] if gemini_preds and trait in gemini_preds else None
            
            # Ensemble strategy: weighted average
            if gemini_score is not None:
                # BERT weight: 0.7 (trained on your data)
                # Gemini weight: 0.3 (general reasoning)
                ensemble_score = round(0.7 * bert_score + 0.3 * gemini_score)
            else:
                ensemble_score = bert_score
            
            ensemble_preds[trait] = {
                'bert_score': bert_score,
                'gemini_score': gemini_score,
                'ensemble_score': ensemble_score,
                'confidence': bert_preds[trait]['confidence']
            }
        
        return ensemble_preds

# ============================================================================
# INITIALIZE API
# ============================================================================

bert_predictor = None
ensemble_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context: load BERT models and ensemble predictor"""
    global bert_predictor, ensemble_predictor
    logger.info("🚀 Starting AI4Life Combined API...")

    try:
        bert_predictor = BERTPredictor(MODEL_DIR, DEVICE)
        ensemble_predictor = EnsemblePredictor(bert_predictor)
        logger.info("✅ API ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield  # Control returns to FastAPI here

    # Optional: cleanup code on shutdown
    logger.info("🛑 Shutting down AI4Life API...")
    
app = FastAPI(
    title="AI4Life Combined Assessment API",
    description="Multi-trait assessment using BERT + Gemini ensemble",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AI4Life Combined Assessment API",
        "status": "running",
        "methods": ["BERT", "Gemini", "Ensemble"],
        "device": str(DEVICE),
        "gemini_enabled": gemini_client is not None
    }

@app.get("/health")
async def health_check():
    if bert_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "bert_models": len(bert_predictor.models),
        "gemini_enabled": gemini_client is not None,
        "device": str(DEVICE)
    }

@app.post("/assess", response_model=AssessmentResponse)
async def assess_interview(request: AssessmentRequest):
    """
    Assess interview using BERT + Gemini ensemble
    
    Returns combined scores from both models for more robust assessment
    """
    try:
        if len(request.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text too short")
        
        # Get ensemble predictions
        predictions = ensemble_predictor.predict(
            request.text, 
            request.include_confidence,
            request.use_ensemble
        )
        
        # Build trait scores
        trait_scores = []
        bert_total = 0
        gemini_total = 0
        gemini_count = 0
        ensemble_weighted_total = 0
        total_weight = 0
        
        for trait, pred in predictions.items():
            score_obj = TraitScore(
                trait=TRAIT_NAMES[trait],
                bert_score=pred['bert_score'],
                gemini_score=pred['gemini_score'],
                ensemble_score=pred['ensemble_score'],
                priority=TRAIT_PRIORITIES[trait],
                confidence=pred['confidence']
            )
            trait_scores.append(score_obj)
            
            # Calculate averages
            bert_total += pred['bert_score']
            
            if pred['gemini_score'] is not None:
                gemini_total += pred['gemini_score']
                gemini_count += 1
            
            # Weighted ensemble (higher priority = higher weight)
            weight = 5 - TRAIT_PRIORITIES[trait]
            ensemble_weighted_total += pred['ensemble_score'] * weight
            total_weight += weight
        
        # Overall scores
        bert_overall = round(bert_total / 4, 2)
        gemini_overall = round(gemini_total / gemini_count, 2) if gemini_count > 0 else None
        overall_score = round(ensemble_weighted_total / total_weight, 2)
        
        # Recommendation
        if overall_score >= 4.0:
            recommendation = "Strong candidate - Recommend for hire"
        elif overall_score >= 3.5:
            recommendation = "Good candidate - Proceed to next round"
        elif overall_score >= 3.0:
            recommendation = "Average candidate - Consider with caution"
        else:
            recommendation = "Weak candidate - Not recommended"
        
        # Determine method used
        if gemini_overall is not None:
            method = "Ensemble (BERT 70% + Gemini 30%)"
        else:
            method = "BERT only (Gemini unavailable)"
        
        # Sort by priority
        trait_scores.sort(key=lambda x: x.priority)
        
        return AssessmentResponse(
            overall_score=overall_score,
            bert_overall=bert_overall,
            gemini_overall=gemini_overall,
            trait_scores=trait_scores,
            recommendation=recommendation,
            method_used=method,
            text_length=len(request.text)
        )
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess-bert-only")
async def assess_bert_only(request: AssessmentRequest):
    """Assessment using BERT models only (faster)"""
    request.use_ensemble = False
    return await assess_interview(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000, log_level="info")