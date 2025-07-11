# 📡 API Documentation

## Base URL
- **Local Development**: `http://localhost:8000`
- **Docker Compose**: `http://localhost:8000`

## Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Authentication
No authentication required for this demo application.

## Endpoints

### 1. Root Endpoint
```http
GET /
```

**Description**: Basic API information and status

**Response**:
```json
{
  "message": "Sentiment Analysis API",
  "status": "running"
}
```

**Status Codes**:
- `200 OK`: API is running

---

### 2. Health Check
```http
GET /health
```

**Description**: Detailed health status including model information

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "quantized": false,
  "onnx_available": false,
  "model_load_time": 5.71,
  "device": "cpu",
  "timestamp": 1752209912.117423
}
```

**Response Fields**:
- `status`: Overall health status ("healthy" | "unhealthy")
- `model_loaded`: Whether the ML model is loaded
- `quantized`: Whether using quantized model
- `onnx_available`: Whether ONNX runtime is available
- `model_load_time`: Time taken to load model (seconds)
- `device`: Device used for inference ("cpu" | "cuda")
- `timestamp`: Unix timestamp of response

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

---

### 3. Sentiment Prediction
```http
POST /predict
```

**Description**: Analyze sentiment of provided text

**Request Body**:
```json
{
  "text": "I love this amazing product!"
}
```

**Request Schema**:
- `text` (string, required): Text to analyze (max length: ~512 tokens)

**Response**:
```json
{
  "label": "positive",
  "score": 0.8945
}
```

**Response Schema**:
- `label` (string): Predicted sentiment ("positive" | "negative")
- `score` (float): Confidence score (0.0 to 1.0)

**Status Codes**:
- `200 OK`: Successful prediction
- `422 Unprocessable Entity`: Invalid request body
- `500 Internal Server Error`: Model inference error

**Example Requests**:

```bash
# Positive sentiment
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this product!"}'

# Negative sentiment
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is terrible and disappointing."}'
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error description"
}
```

### Common Errors

**422 Unprocessable Entity**:
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Model inference failed"
}
```

## Rate Limiting
No rate limiting implemented in this demo version.

## CORS
CORS is enabled for:
- `http://localhost:3000` (Docker frontend)
- `http://localhost:5173` (Vite dev server)

## Model Information
- **Base Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Task**: Binary sentiment classification
- **Input**: Text (English)
- **Output**: Positive/Negative with confidence score
- **Fine-tuning**: Supports custom dataset fine-tuning

## Performance
- **Average Response Time**: ~2.1 seconds
- **Model Loading Time**: ~5.7 seconds (startup)
- **Concurrent Requests**: Supported
- **Memory Usage**: ~2-3GB RAM
