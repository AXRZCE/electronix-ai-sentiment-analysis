from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
from typing import Dict, Any, Union
import time

# Try to import ONNX runtime components
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ORTModelForSequenceClassification = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_quantized = False
model_load_time = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

def load_model():
    """Load the sentiment analysis model and tokenizer with quantization support"""
    global model, tokenizer, is_quantized, model_load_time

    start_time = time.time()
    # Check for local model directory (for development) or use current directory (for Docker)
    model_path = "./model" if os.path.exists("./model") else "../model"
    quantized_path = "./model_quantized" if os.path.exists("./model_quantized") else "../model_quantized"
    use_quantized = os.getenv("QUANTIZED_MODEL", "false").lower() == "true"

    try:
        # Priority: Quantized -> Fine-tuned -> Pre-trained
        if use_quantized and ONNX_AVAILABLE and os.path.exists(quantized_path):
            logger.info(f"Loading quantized model from {quantized_path}")

            # Check for quantized ONNX model
            quantized_model_file = os.path.join(quantized_path, "model_quantized.onnx")
            if os.path.exists(quantized_model_file):
                model = ORTModelForSequenceClassification.from_pretrained(
                    quantized_path,
                    file_name="model_quantized.onnx"
                )
                tokenizer = AutoTokenizer.from_pretrained(quantized_path)
                is_quantized = True
                logger.info("Quantized ONNX model loaded successfully")
            else:
                # Fall back to regular ONNX model
                model = ORTModelForSequenceClassification.from_pretrained(quantized_path)
                tokenizer = AutoTokenizer.from_pretrained(quantized_path)
                is_quantized = False
                logger.info("Regular ONNX model loaded successfully")

        elif os.path.exists(model_path) and os.listdir(model_path):
            # Load fine-tuned PyTorch model
            logger.info(f"Loading fine-tuned PyTorch model from {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.to(device)
            is_quantized = False

        else:
            # Load pre-trained model (using smaller model for memory efficiency)
            model_name = os.getenv("MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
            logger.info(f"Loading pre-trained model: {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.to(device)
            is_quantized = False

        if not is_quantized:
            model.eval()

        model_load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {model_load_time:.2f}s (quantized: {is_quantized})")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fall back to PyTorch model if ONNX fails
        if use_quantized:
            logger.info("Falling back to PyTorch model...")
            try:
                if os.path.exists(model_path) and os.listdir(model_path):
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                model.to(device)
                model.eval()
                is_quantized = False
                model_load_time = time.time() - start_time
                logger.info(f"Fallback model loaded successfully in {model_load_time:.2f}s")
            except Exception as fallback_error:
                logger.error(f"Fallback model loading failed: {fallback_error}")
                raise fallback_error
        else:
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "quantized": is_quantized,
        "onnx_available": ONNX_AVAILABLE,
        "model_load_time": model_load_time,
        "device": str(device) if not is_quantized else "cpu (ONNX)",
        "timestamp": time.time()
    }

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
        
        # Get prediction (handle both PyTorch and ONNX models)
        if is_quantized:
            # ONNX model inference
            outputs = model(**inputs)
            logits = outputs.logits
            if isinstance(logits, torch.Tensor):
                predictions = torch.nn.functional.softmax(logits, dim=-1)
            else:
                # Convert numpy to torch if needed
                import numpy as np
                logits = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
        else:
            # PyTorch model inference
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
