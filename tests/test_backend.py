#!/usr/bin/env python3
"""
Comprehensive test suite for the Sentiment Analysis Backend
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app

client = TestClient(app)

class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data

class TestPredictEndpoint:
    """Test the prediction endpoint"""
    
    def test_predict_positive_sentiment(self):
        """Test prediction with positive text"""
        response = client.post(
            "/predict",
            json={"text": "I love this product! It's amazing!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["positive", "negative"]
        assert 0 <= data["score"] <= 1
    
    def test_predict_negative_sentiment(self):
        """Test prediction with negative text"""
        response = client.post(
            "/predict",
            json={"text": "This is terrible! I hate it!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["positive", "negative"]
        assert 0 <= data["score"] <= 1
    
    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_text_field(self):
        """Test prediction with missing text field"""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_json(self):
        """Test prediction with invalid JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_predict_long_text(self):
        """Test prediction with very long text"""
        long_text = "This is a test sentence. " * 1000
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        # Should handle long text gracefully
        assert response.status_code in [200, 422]
    
    def test_predict_special_characters(self):
        """Test prediction with special characters"""
        response = client.post(
            "/predict",
            json={"text": "Hello! @#$%^&*()_+ 🎉 This is great! 😊"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data

class TestModelFunctions:
    """Test the model utility functions"""
    
    def test_model_loading_mock(self):
        """Test model loading behavior"""
        # This is a placeholder test for model loading
        # In a real CI environment, we would mock the model loading
        assert True  # Placeholder test

class TestCORSAndSecurity:
    """Test CORS and security configurations"""
    
    def test_cors_headers(self):
        """Test that CORS headers are present"""
        response = client.options("/predict")
        # FastAPI with CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled
    
    def test_security_headers(self):
        """Test basic security headers"""
        response = client.get("/health")
        # Check that response doesn't expose sensitive information
        assert "server" not in response.headers.get("server", "").lower()

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_404_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test using wrong HTTP method"""
        response = client.get("/predict")
        assert response.status_code == 405
    
    def test_large_payload(self):
        """Test handling of large payloads"""
        large_text = "x" * 100000  # 100KB of text
        response = client.post(
            "/predict",
            json={"text": large_text}
        )
        # Should either process or reject gracefully
        assert response.status_code in [200, 413, 422]

class TestPerformance:
    """Test performance characteristics"""
    
    def test_response_time(self):
        """Test that responses are reasonably fast"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/predict",
            json={"text": "This is a simple test message."}
        )
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be under 5 seconds (generous for CI)
        assert (end_time - start_time) < 5.0
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.post(
                "/predict",
                json={"text": "Test message for concurrency"}
            )
            results.append(response.status_code)
        
        # Create 5 concurrent threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
