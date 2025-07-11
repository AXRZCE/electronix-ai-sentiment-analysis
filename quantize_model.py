#!/usr/bin/env python3
"""
Model quantization script for sentiment analysis model
Converts PyTorch models to optimized ONNX format with quantization
Usage: python quantize_model.py --model_path ./model --output_dir ./model_quantized
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pytorch_model(model_path: str):
    """Load PyTorch model and tokenizer"""
    logger.info(f"Loading PyTorch model from {model_path}")
    
    try:
        if os.path.exists(model_path) and os.listdir(model_path):
            # Load fine-tuned model
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Loaded fine-tuned model")
        else:
            # Load pre-trained model
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded pre-trained model: {model_name}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def convert_to_onnx(model, tokenizer, output_dir: str, optimize: bool = True):
    """Convert PyTorch model to ONNX format"""
    logger.info("Converting PyTorch model to ONNX...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to ONNX
    ort_model = ORTModelForSequenceClassification.from_transformers(
        model=model,
        tokenizer=tokenizer,
        save_dir=output_dir,
        file_name="model.onnx"
    )
    
    if optimize:
        logger.info("Optimizing ONNX model...")
        
        # Create optimization configuration
        optimization_config = AutoOptimizationConfig.with_optimization_level(
            optimization_level="O2",  # Aggressive optimization
            optimize_for_gpu=False,   # CPU optimization
            fp16=False               # Keep FP32 for better compatibility
        )
        
        # Optimize the model
        optimizer = ORTOptimizer.from_pretrained(output_dir)
        optimizer.optimize(
            save_dir=output_dir,
            optimization_config=optimization_config,
            file_suffix="optimized"
        )
        
        logger.info("ONNX model optimized")
    
    return ort_model

def quantize_model(model_dir: str, quantization_approach: str = "dynamic"):
    """Quantize ONNX model for faster inference"""
    logger.info(f"Quantizing model with {quantization_approach} quantization...")
    
    # Create quantization configuration
    if quantization_approach == "dynamic":
        quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False)
    elif quantization_approach == "static":
        quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=True)
    else:
        raise ValueError(f"Unsupported quantization approach: {quantization_approach}")
    
    # Load the ONNX model
    quantizer = ORTQuantizer.from_pretrained(model_dir)
    
    # Quantize the model
    quantizer.quantize(
        save_dir=model_dir,
        quantization_config=quantization_config,
        file_suffix="quantized"
    )
    
    logger.info("Model quantized successfully")

def benchmark_models(original_dir: str, quantized_dir: str, test_texts: list):
    """Benchmark original vs quantized model performance"""
    logger.info("Benchmarking model performance...")
    
    # Load models
    original_model = ORTModelForSequenceClassification.from_pretrained(original_dir)
    original_tokenizer = AutoTokenizer.from_pretrained(original_dir)
    
    quantized_model = ORTModelForSequenceClassification.from_pretrained(
        quantized_dir, 
        file_name="model_quantized.onnx"
    )
    quantized_tokenizer = AutoTokenizer.from_pretrained(quantized_dir)
    
    def benchmark_model(model, tokenizer, name):
        """Benchmark a single model"""
        times = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"{name} - Average inference time: {avg_time:.4f}s (±{std_time:.4f}s)")
        return avg_time, std_time
    
    # Benchmark both models
    orig_time, orig_std = benchmark_model(original_model, original_tokenizer, "Original Model")
    quant_time, quant_std = benchmark_model(quantized_model, quantized_tokenizer, "Quantized Model")
    
    # Calculate speedup
    speedup = orig_time / quant_time
    logger.info(f"Quantized model is {speedup:.2f}x faster")
    
    return {
        "original": {"time": orig_time, "std": orig_std},
        "quantized": {"time": quant_time, "std": quant_std},
        "speedup": speedup
    }

def get_model_size(model_dir: str):
    """Get model file sizes"""
    sizes = {}
    
    for file_path in Path(model_dir).glob("*.onnx"):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        sizes[file_path.name] = size_mb
        logger.info(f"{file_path.name}: {size_mb:.2f} MB")
    
    return sizes

def main():
    parser = argparse.ArgumentParser(description='Quantize sentiment analysis model')
    parser.add_argument('--model_path', default='./model', 
                       help='Path to PyTorch model directory')
    parser.add_argument('--output_dir', default='./model_quantized', 
                       help='Output directory for quantized model')
    parser.add_argument('--quantization', choices=['dynamic', 'static'], default='dynamic',
                       help='Quantization approach')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Apply ONNX optimizations')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Benchmark original vs quantized model')
    
    args = parser.parse_args()
    
    # Test texts for benchmarking
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "The movie was okay, nothing special.",
        "Absolutely fantastic experience!",
        "Could be better, but not bad.",
        "Worst purchase ever made.",
        "Highly recommend this to everyone!",
        "Not worth the money at all.",
        "Pretty good overall quality.",
        "Disappointed with the service."
    ]
    
    try:
        # Load original PyTorch model
        pytorch_model, tokenizer = load_pytorch_model(args.model_path)
        
        # Convert to ONNX
        ort_model = convert_to_onnx(pytorch_model, tokenizer, args.output_dir, args.optimize)
        
        # Quantize the model
        quantize_model(args.output_dir, args.quantization)
        
        # Get model sizes
        logger.info("Model sizes:")
        sizes = get_model_size(args.output_dir)
        
        # Benchmark if requested
        if args.benchmark:
            benchmark_results = benchmark_models(args.output_dir, args.output_dir, test_texts)
            
            # Save benchmark results
            import json
            with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
                json.dump({
                    "model_sizes": sizes,
                    "performance": benchmark_results
                }, f, indent=2)
        
        logger.info(f"Quantization complete! Quantized model saved to {args.output_dir}")
        logger.info("To use the quantized model, set QUANTIZED_MODEL=true in your environment")
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise

if __name__ == "__main__":
    main()
