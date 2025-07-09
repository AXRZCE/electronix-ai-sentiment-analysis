# 🎭 Electronix AI - Sentiment Analysis Microservice

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)

> **Complete end-to-end microservice for binary sentiment analysis with modern web frontend**

A complete end-to-end microservice for binary sentiment analysis with a modern web frontend, built with Python FastAPI backend and React TypeScript frontend.

## 🚀 Features

- **Binary Sentiment Analysis**: Classify text as positive or negative with confidence scores
- **Pre-trained Model**: Uses Hugging Face Transformers with cardiffnlp/twitter-roberta-base-sentiment-latest
- **Fine-tuning Support**: CLI script to fine-tune on custom datasets
- **Modern Frontend**: React TypeScript with beautiful UI and real-time predictions
- **Containerized**: Docker Compose setup with optional GPU support
- **Production Ready**: Health checks, error handling, and proper logging

## 📋 Requirements

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for local frontend development)
- NVIDIA Docker (optional, for GPU support)

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   React Frontend│ ──────────────► │  FastAPI Backend│
│   (Port 3000)   │                 │   (Port 8000)   │
└─────────────────┘                 └─────────────────┘
                                            │
                                            ▼
                                    ┌─────────────────┐
                                    │ Hugging Face    │
                                    │ Transformers    │
                                    │ Model           │
                                    └─────────────────┘
```

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project directory**
2. **Start the application:**

   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### With GPU Support

```bash
docker-compose --profile gpu up --build
```

## 🔧 Local Development

### Backend Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the backend:**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Install dependencies:**

   ```bash
   cd frontend
   npm install
   ```

2. **Run the development server:**
   ```bash
   npm run dev
   ```

## 🤖 Fine-tuning

### Data Format

Create a JSONL file with one JSON object per line:

```json
{"text": "Great product! I love it.", "label": "positive"}
{"text": "Terrible quality, waste of money.", "label": "negative"}
```

### Fine-tuning Command

```bash
python finetune.py --data data/sample_data.jsonl --epochs 3 --lr 3e-5
```

### Fine-tuning Options

- `--data`: Path to training data (JSONL format)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 3e-5)
- `--batch_size`: Batch size (default: 16)
- `--max_length`: Maximum sequence length (default: 512)
- `--model_name`: Pre-trained model name
- `--output_dir`: Output directory for fine-tuned model (default: ./model)
- `--seed`: Random seed for reproducibility (default: 42)
- `--validation_split`: Validation split ratio (default: 0.2)

### Fine-tuning Features

- **Cross-entropy loss** with proper gradient clipping
- **Learning rate scheduler** with warmup
- **Deterministic training** with fixed seeds
- **Automatic validation** with best model saving
- **Progress tracking** with detailed metrics

## 📡 API Documentation

### Endpoints

#### `GET /`

- **Description**: Root endpoint
- **Response**: `{"message": "Sentiment Analysis API", "status": "running"}`

#### `GET /health`

- **Description**: Health check endpoint
- **Response**: `{"status": "healthy", "model_loaded": true}`

#### `POST /predict`

- **Description**: Predict sentiment for given text
- **Request Body**:
  ```json
  {
    "text": "Your text here"
  }
  ```
- **Response**:
  ```json
  {
    "label": "positive",
    "score": 0.8945
  }
  ```

### Response Format

All prediction responses follow this format:

- `label`: Either "positive" or "negative"
- `score`: Confidence score between 0 and 1

## 🎨 Design Decisions

### Backend Architecture

- **FastAPI**: Chosen for automatic API documentation, type hints, and async support
- **Hugging Face Transformers**: Industry-standard library for transformer models
- **PyTorch**: Flexible deep learning framework with excellent model support
- **Automatic Model Loading**: Checks for fine-tuned models first, falls back to pre-trained

### Frontend Architecture

- **React with TypeScript**: Type safety and modern development experience
- **Vite**: Fast build tool with hot module replacement
- **Vanilla CSS**: Custom styling for better performance and control
- **Responsive Design**: Mobile-friendly interface with gradient backgrounds

### Containerization

- **Multi-stage builds**: Optimized Docker images for production
- **Health checks**: Proper service monitoring and dependency management
- **Volume mounts**: Persistent model and data storage
- **Network isolation**: Secure service communication

### Fine-tuning Implementation

- **Reproducible training**: Fixed seeds and deterministic operations
- **Proper validation**: Automatic train/validation split with best model saving
- **Gradient clipping**: Prevents exploding gradients during training
- **Learning rate scheduling**: Warmup and linear decay for stable training

## 🔍 Performance Notes

### CPU Training Time (Approximate)

- **Small dataset (100-1000 samples)**: 2-5 minutes per epoch
- **Medium dataset (1000-10000 samples)**: 10-30 minutes per epoch
- **Large dataset (10000+ samples)**: 30+ minutes per epoch

### GPU Training Time (Approximate)

- **Small dataset**: 30-60 seconds per epoch
- **Medium dataset**: 2-5 minutes per epoch
- **Large dataset**: 5-15 minutes per epoch

_Note: Times vary based on hardware specifications and model complexity_

## 🐛 Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure the model directory exists and contains valid model files
2. **CORS errors**: Frontend and backend must be on the same network (handled in Docker Compose)
3. **Memory issues**: Reduce batch size or use gradient accumulation for large datasets
4. **GPU not detected**: Ensure NVIDIA Docker is properly installed and configured

### Logs

- **Backend logs**: `docker-compose logs backend`
- **Frontend logs**: `docker-compose logs frontend`
- **All logs**: `docker-compose logs -f`

## 📦 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── Dockerfile           # Backend container config
│   └── Dockerfile.gpu       # GPU-enabled container config
├── frontend/
│   ├── src/
│   │   ├── main.ts         # Main application logic
│   │   └── style.css       # Custom styles
│   ├── Dockerfile          # Frontend container config
│   └── nginx.conf          # Nginx configuration
├── data/
│   └── sample_data.jsonl   # Sample training data
├── model/                  # Fine-tuned model storage
├── finetune.py            # Fine-tuning script
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Container orchestration
└── README.md             # This file
```

## 🎬 Demo Video

_Demo video will be uploaded to YouTube and link provided here_

## 📄 License

This project is created for educational purposes as part of the Electronix AI assignment.

---

**Built with ❤️ using FastAPI, React, and Hugging Face Transformers**
