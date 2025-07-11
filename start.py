#!/usr/bin/env python3
"""
Startup script for Render deployment
This script helps debug and start the application properly on Render
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check the deployment environment"""
    logger.info("=== Environment Check ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Check for model directory
    model_paths = ["./model", "../model", "./backend/../model"]
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"Found model directory at: {path}")
            logger.info(f"Model files: {os.listdir(path)}")
            break
    else:
        logger.warning("No model directory found - will use pre-trained model")
    
    # Check Python packages
    try:
        import torch
        import transformers
        import fastapi
        logger.info("✅ All required packages imported successfully")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"FastAPI version: {fastapi.__version__}")
    except ImportError as e:
        logger.error(f"❌ Package import error: {e}")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    logger.info("=== Starting Server ===")
    
    # Get port from environment (Render sets this)
    port = os.environ.get("PORT", "8000")
    
    # Change to backend directory
    os.chdir("backend")
    logger.info(f"Changed to directory: {os.getcwd()}")
    
    # Start uvicorn
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", port,
        "--log-level", "info"
    ]
    
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    logger.info("🚀 Starting Render deployment...")
    
    if check_environment():
        start_server()
    else:
        logger.error("❌ Environment check failed")
        sys.exit(1)
