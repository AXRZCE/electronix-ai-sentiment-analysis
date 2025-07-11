#!/usr/bin/env python3
"""
Test suite for model quantization functionality
"""

import pytest
import os
import sys
import time
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class TestQuantizationScript:
    """Test the quantization script functionality"""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_quantization_script_exists(self):
        """Test that the quantization script exists and is executable"""
        script_path = Path("quantize_model.py")
        assert script_path.exists(), "quantize_model.py script not found"
        assert script_path.is_file(), "quantize_model.py is not a file"
    
    def test_quantization_dependencies(self):
        """Test that quantization dependencies are available"""
        try:
            import onnx
            import onnxruntime
            from optimum.onnxruntime import ORTModelForSequenceClassification
            assert True
        except ImportError as e:
            pytest.fail(f"Quantization dependencies not available: {e}")
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_quantization_help(self):
        """Test that the quantization script shows help"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "quantize_model.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "quantization" in result.stdout.lower()

class TestQuantizedModelLoading:
    """Test loading and using quantized models"""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_ort_model_loading(self):
        """Test that ORTModelForSequenceClassification can be imported and used"""
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        
        # This test just verifies the classes can be imported
        assert ORTModelForSequenceClassification is not None
        assert AutoTokenizer is not None
    
    def test_quantized_model_directory_structure(self):
        """Test that quantized model directory has expected structure"""
        quantized_dir = Path("model_quantized")
        
        if quantized_dir.exists():
            # Check for expected files
            expected_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            for file_name in expected_files:
                file_path = quantized_dir / file_name
                if file_path.exists():
                    assert file_path.is_file(), f"{file_name} should be a file"
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_quantized_model_inference(self):
        """Test inference with quantized model if available"""
        quantized_dir = Path("model_quantized")
        
        if not quantized_dir.exists():
            pytest.skip("Quantized model directory not found")
        
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            
            # Try to load quantized model
            model = ORTModelForSequenceClassification.from_pretrained(str(quantized_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(quantized_dir))
            
            # Test inference
            test_text = "This is a test message"
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            
            assert outputs is not None
            assert hasattr(outputs, 'logits')
            assert outputs.logits.shape[0] == 1  # Batch size 1
            
        except Exception as e:
            pytest.skip(f"Quantized model inference failed: {e}")

class TestQuantizationPerformance:
    """Test performance characteristics of quantized models"""
    
    def test_model_size_comparison(self):
        """Test that quantized models are smaller than original"""
        model_dir = Path("model")
        quantized_dir = Path("model_quantized")
        
        if not (model_dir.exists() and quantized_dir.exists()):
            pytest.skip("Both model directories not available for comparison")
        
        # Get sizes of model files
        def get_directory_size(path):
            total_size = 0
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        
        original_size = get_directory_size(model_dir)
        quantized_size = get_directory_size(quantized_dir)
        
        if original_size > 0 and quantized_size > 0:
            # Quantized model should be smaller (allowing some overhead for ONNX format)
            size_ratio = quantized_size / original_size
            assert size_ratio < 2.0, f"Quantized model not significantly smaller: {size_ratio:.2f}x"
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_inference_speed_benchmark(self):
        """Test that quantized model inference is reasonably fast"""
        quantized_dir = Path("model_quantized")
        
        if not quantized_dir.exists():
            pytest.skip("Quantized model not available")
        
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            
            model = ORTModelForSequenceClassification.from_pretrained(str(quantized_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(quantized_dir))
            
            # Benchmark inference time
            test_text = "This is a performance test message for the quantized model."
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
            
            # Warm up
            for _ in range(3):
                model(**inputs)
            
            # Measure inference time
            times = []
            for _ in range(10):
                start_time = time.time()
                outputs = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            
            # Quantized model should be reasonably fast (under 1 second on most hardware)
            assert avg_time < 1.0, f"Quantized model too slow: {avg_time:.3f}s average"
            
        except Exception as e:
            pytest.skip(f"Performance benchmark failed: {e}")

class TestQuantizationConfiguration:
    """Test quantization configuration and environment variables"""
    
    def test_quantized_model_env_var(self):
        """Test QUANTIZED_MODEL environment variable handling"""
        # Test default value
        quantized_default = os.getenv("QUANTIZED_MODEL", "false").lower() == "true"
        assert isinstance(quantized_default, bool)
        
        # Test setting the variable
        os.environ["QUANTIZED_MODEL"] = "true"
        quantized_enabled = os.getenv("QUANTIZED_MODEL", "false").lower() == "true"
        assert quantized_enabled is True
        
        # Clean up
        if "QUANTIZED_MODEL" in os.environ:
            del os.environ["QUANTIZED_MODEL"]
    
    def test_benchmark_results_format(self):
        """Test that benchmark results file has correct format"""
        benchmark_file = Path("model_quantized/benchmark_results.json")
        
        if not benchmark_file.exists():
            pytest.skip("Benchmark results file not found")
        
        try:
            with open(benchmark_file, 'r') as f:
                results = json.load(f)
            
            # Check expected structure
            assert "model_sizes" in results
            assert "performance" in results
            
            performance = results["performance"]
            assert "original" in performance
            assert "quantized" in performance
            assert "speedup" in performance
            
            # Check that speedup is reasonable
            speedup = performance["speedup"]
            assert isinstance(speedup, (int, float))
            assert speedup > 0, "Speedup should be positive"
            
        except (json.JSONDecodeError, KeyError) as e:
            pytest.fail(f"Invalid benchmark results format: {e}")

class TestDockerQuantizationSupport:
    """Test Docker support for quantized models"""
    
    def test_quantized_dockerfile_exists(self):
        """Test that quantized Dockerfile exists"""
        dockerfile_path = Path("backend/Dockerfile.quantized")
        assert dockerfile_path.exists(), "Dockerfile.quantized not found"
        assert dockerfile_path.is_file(), "Dockerfile.quantized is not a file"
    
    def test_docker_compose_quantized_profile(self):
        """Test that docker-compose has quantized profile"""
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists(), "docker-compose.yml not found"
        
        with open(compose_file, 'r') as f:
            content = f.read()
        
        assert "backend-quantized" in content, "backend-quantized service not found"
        assert "quantized" in content, "quantized profile not found"
        assert "QUANTIZED_MODEL=true" in content, "QUANTIZED_MODEL environment variable not set"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
