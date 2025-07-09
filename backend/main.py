from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

def load_model():
    """Load the sentiment analysis model and tokenizer"""
    global model, tokenizer
    
    model_path = "./model"
    
    try:
        # Check if fine-tuned model exists
        if os.path.exists(model_path) and os.listdir(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Load pre-trained model
            logger.info("Loading pre-trained model: cardiffnlp/twitter-roberta-base-sentiment-latest")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput) -> PredictionResponse:
    """Predict sentiment for given text"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            input_data.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get the predicted class and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Map class to label (assuming binary classification: 0=negative, 1=positive)
        # For cardiffnlp model: 0=negative, 1=neutral, 2=positive
        if model.config.num_labels == 3:
            label_map = {0: "negative", 1: "negative", 2: "positive"}  # Treat neutral as negative for binary
        else:
            label_map = {0: "negative", 1: "positive"}
        
        label = label_map.get(predicted_class, "unknown")
        
        return PredictionResponse(label=label, score=confidence)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
