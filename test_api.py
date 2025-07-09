#!/usr/bin/env python3
"""
Simple test script for the Sentiment Analysis API
Usage: python test_api.py
"""

import requests
import json
import time

# API Configuration
BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"

# Test cases
TEST_CASES = [
    {
        "text": "I love this product! It's amazing and works perfectly.",
        "expected": "positive"
    },
    {
        "text": "This is terrible quality and completely useless.",
        "expected": "negative"
    },
    {
        "text": "Great service and fast delivery. Highly recommend!",
        "expected": "positive"
    },
    {
        "text": "Worst experience ever. Don't waste your money.",
        "expected": "negative"
    },
    {
        "text": "The product is okay, nothing special but works fine.",
        "expected": "positive"  # Neutral-positive
    }
]

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction(text, expected=None):
    """Test a single prediction"""
    print(f"\n🧪 Testing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    try:
        payload = {"text": text}
        response = requests.post(
            PREDICT_ENDPOINT, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            label = data.get("label")
            score = data.get("score")
            
            print(f"📊 Result: {label.upper()} (confidence: {score:.3f})")
            
            if expected and label == expected:
                print("✅ Prediction matches expected result")
            elif expected:
                print(f"⚠️  Expected {expected}, got {label}")
            
            return True, data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False, None

def test_api_performance():
    """Test API response time"""
    print("\n⏱️  Testing API performance...")
    
    test_text = "This is a simple test message for performance testing."
    times = []
    
    for i in range(5):
        start_time = time.time()
        success, _ = test_prediction(test_text)
        end_time = time.time()
        
        if success:
            response_time = end_time - start_time
            times.append(response_time)
            print(f"Request {i+1}: {response_time:.3f}s")
        else:
            print(f"Request {i+1}: Failed")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📈 Average response time: {avg_time:.3f}s")
        print(f"📈 Min response time: {min(times):.3f}s")
        print(f"📈 Max response time: {max(times):.3f}s")

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🚨 Testing error handling...")
    
    # Test empty text
    print("Testing empty text...")
    success, _ = test_prediction("")
    
    # Test very long text
    print("Testing very long text...")
    long_text = "This is a test. " * 1000  # Very long text
    success, _ = test_prediction(long_text)
    
    # Test invalid JSON (this will be handled by requests)
    print("Testing malformed request...")
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Malformed request response: {response.status_code}")
    except Exception as e:
        print(f"Malformed request error: {e}")

def main():
    """Run all tests"""
    print("🎭 Sentiment Analysis API Test Suite")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("\n❌ Health check failed. Make sure the backend is running.")
        print("Start with: docker-compose up --build")
        return
    
    print("\n" + "=" * 50)
    print("🧪 Running prediction tests...")
    
    # Test all cases
    passed = 0
    total = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n--- Test Case {i}/{total} ---")
        success, result = test_prediction(test_case["text"], test_case["expected"])
        if success:
            passed += 1
    
    # Performance testing
    test_api_performance()
    
    # Error handling testing
    test_error_handling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"✅ Passed: {passed}/{total} prediction tests")
    print(f"🎯 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
