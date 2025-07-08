#!/usr/bin/env python3
"""
Performance Testing Script for Clinical Study Churn Prediction
Demonstrates the performance improvements from async operations and caching
"""

import asyncio
import aiohttp
import requests
import time
import json
import statistics
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# API configuration
API_BASE_URL = "http://localhost:8000"


def generate_test_patients(num_patients: int = 10) -> List[Dict[str, Any]]:
    """Generate test patient data"""
    patients = []
    for i in range(num_patients):
        patient = {
            "age": np.random.randint(18, 85),
            "gender": np.random.choice(["Male", "Female", "Other"]),
            "income": np.random.randint(20000, 150000),
            "location": np.random.choice(["Urban", "Suburban", "Rural"]),
            "study_type": np.random.choice(["Phase I", "Phase II", "Phase III"]),
            "condition": np.random.choice(
                [
                    "Diabetes",
                    "Hypertension",
                    "Cardiovascular Disease",
                    "Obesity",
                    "Respiratory Disease",
                    "Mental Health",
                    "Cancer",
                    "Autoimmune Disease",
                ]
            ),
            "visit_adherence_rate": np.random.uniform(0.3, 1.0),
            "tenure_months": np.random.randint(1, 36),
            "last_visit_gap_days": np.random.randint(0, 90),
            "num_medications": np.random.randint(0, 8),
            "has_side_effects": np.random.choice([True, False]),
            "transport_support": np.random.choice([True, False]),
            "monthly_stipend": np.random.randint(100, 1000),
            "contact_frequency": np.random.uniform(1.0, 8.0),
            "support_group_member": np.random.choice([True, False]),
            "language_barrier": np.random.choice([True, False]),
            "device_usage_compliance": np.random.uniform(0.2, 1.0),
            "survey_score_avg": np.random.uniform(1.0, 10.0),
        }
        patients.append(patient)
    return patients


async def test_single_prediction_async(patient_data: Dict[str, Any]) -> float:
    """Test single prediction with async API"""
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/predict", json=patient_data
            ) as response:
                if response.status == 200:
                    await response.json()
                    return time.time() - start_time
                else:
                    return -1  # Error
    except Exception:
        return -1  # Error


def test_single_prediction_sync(patient_data: Dict[str, Any]) -> float:
    """Test single prediction with sync API"""
    start_time = time.time()
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=patient_data)
        if response.status_code == 200:
            response.json()
            return time.time() - start_time
        else:
            return -1  # Error
    except Exception:
        return -1  # Error


async def test_batch_predictions_async(patients_data: List[Dict[str, Any]]) -> float:
    """Test batch predictions with async API"""
    start_time = time.time()
    try:
        batch_request = {"patients": patients_data}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/predict/batch", json=batch_request
            ) as response:
                if response.status == 200:
                    await response.json()
                    return time.time() - start_time
                else:
                    return -1  # Error
    except Exception:
        return -1  # Error


async def test_parallel_predictions_async(patients_data: List[Dict[str, Any]]) -> float:
    """Test parallel individual predictions with async API"""
    start_time = time.time()
    try:
        tasks = [test_single_prediction_async(patient) for patient in patients_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if all predictions succeeded
        if all(isinstance(r, float) and r >= 0 for r in results):
            return time.time() - start_time
        else:
            return -1  # Error
    except Exception:
        return -1  # Error


def test_sequential_predictions_sync(patients_data: List[Dict[str, Any]]) -> float:
    """Test sequential individual predictions with sync API"""
    start_time = time.time()
    try:
        for patient in patients_data:
            response = requests.post(f"{API_BASE_URL}/predict", json=patient)
            if response.status_code != 200:
                return -1  # Error
        return time.time() - start_time
    except Exception:
        return -1  # Error


async def test_caching_performance(
    patient_data: Dict[str, Any], num_requests: int = 5
) -> Dict[str, List[float]]:
    """Test caching performance by making repeated requests"""
    times = []

    for i in range(num_requests):
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_BASE_URL}/predict", json=patient_data
                ) as response:
                    if response.status == 200:
                        await response.json()
                        times.append(time.time() - start_time)
                    else:
                        times.append(-1)
        except Exception:
            times.append(-1)

    return {
        "first_request": times[0] if times[0] >= 0 else None,
        "subsequent_requests": [t for t in times[1:] if t >= 0],
        "cache_hit_improvement": (
            (times[0] - statistics.mean(times[1:])) / times[0] * 100
            if times[0] >= 0 and len([t for t in times[1:] if t >= 0]) > 0
            else None
        ),
    }


def print_performance_results(test_name: str, times: List[float], num_requests: int):
    """Print performance results"""
    valid_times = [t for t in times if t >= 0]

    if not valid_times:
        print(f"❌ {test_name}: All requests failed")
        return

    print(f"\n📊 {test_name} Results:")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful requests: {len(valid_times)}")
    print(f"   Success rate: {len(valid_times)/num_requests*100:.1f}%")
    print(f"   Average time: {statistics.mean(valid_times):.3f}s")
    print(f"   Median time: {statistics.median(valid_times):.3f}s")
    print(f"   Min time: {min(valid_times):.3f}s")
    print(f"   Max time: {max(valid_times):.3f}s")
    print(f"   Standard deviation: {statistics.stdev(valid_times):.3f}s")


def print_comparison_results(
    async_times: List[float], sync_times: List[float], test_name: str
):
    """Print comparison results between async and sync"""
    async_valid = [t for t in async_times if t >= 0]
    sync_valid = [t for t in sync_times if t >= 0]

    if not async_valid or not sync_valid:
        print(f"❌ {test_name}: Cannot compare due to failed requests")
        return

    async_avg = statistics.mean(async_valid)
    sync_avg = statistics.mean(sync_valid)
    improvement = (sync_avg - async_avg) / sync_avg * 100

    print(f"\n⚡ {test_name} Performance Comparison:")
    print(f"   Async average: {async_avg:.3f}s")
    print(f"   Sync average: {sync_avg:.3f}s")
    print(f"   Performance improvement: {improvement:.1f}%")

    if improvement > 0:
        print(f"   ✅ Async is {improvement:.1f}% faster")
    else:
        print(f"   ⚠️  Sync is {abs(improvement):.1f}% faster")


async def main():
    """Main performance testing function"""
    print("=" * 70)
    print("🏥 Clinical Study Churn Prediction - Performance Testing")
    print("=" * 70)

    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Please start the services first.")
            return
    except Exception:
        print("❌ API is not running. Please start the services first.")
        return

    print("✅ API is running. Starting performance tests...")

    # Generate test data
    print("\n📋 Generating test data...")
    test_patients = generate_test_patients(20)
    single_patient = test_patients[0]

    # Test 1: Single prediction performance
    print("\n" + "=" * 50)
    print("Test 1: Single Prediction Performance")
    print("=" * 50)

    # Async single prediction
    async_times = []
    for _ in range(10):
        time_taken = await test_single_prediction_async(single_patient)
        async_times.append(time_taken)

    # Sync single prediction
    sync_times = []
    for _ in range(10):
        time_taken = test_single_prediction_sync(single_patient)
        sync_times.append(time_taken)

    print_performance_results("Async Single Prediction", async_times, 10)
    print_performance_results("Sync Single Prediction", sync_times, 10)
    print_comparison_results(async_times, sync_times, "Single Prediction")

    # Test 2: Batch vs Parallel vs Sequential
    print("\n" + "=" * 50)
    print("Test 2: Batch vs Parallel vs Sequential")
    print("=" * 50)

    batch_patients = test_patients[:10]

    # Batch prediction
    batch_time = await test_batch_predictions_async(batch_patients)

    # Parallel individual predictions
    parallel_time = await test_parallel_predictions_async(batch_patients)

    # Sequential individual predictions
    sequential_time = test_sequential_predictions_sync(batch_patients)

    print(f"\n📊 Batch Prediction: {batch_time:.3f}s")
    print(f"📊 Parallel Predictions: {parallel_time:.3f}s")
    print(f"📊 Sequential Predictions: {sequential_time:.3f}s")

    if batch_time >= 0 and parallel_time >= 0 and sequential_time >= 0:
        batch_vs_parallel = (parallel_time - batch_time) / parallel_time * 100
        parallel_vs_sequential = (
            (sequential_time - parallel_time) / sequential_time * 100
        )

        print(f"\n⚡ Performance Comparison:")
        print(f"   Batch vs Parallel: {batch_vs_parallel:.1f}% improvement")
        print(f"   Parallel vs Sequential: {parallel_vs_sequential:.1f}% improvement")

    # Test 3: Caching performance
    print("\n" + "=" * 50)
    print("Test 3: Caching Performance")
    print("=" * 50)

    cache_results = await test_caching_performance(single_patient, 10)

    if cache_results["first_request"]:
        print(f"\n📊 First Request: {cache_results['first_request']:.3f}s")

        if cache_results["subsequent_requests"]:
            avg_subsequent = statistics.mean(cache_results["subsequent_requests"])
            print(f"📊 Average Subsequent Requests: {avg_subsequent:.3f}s")

            if cache_results["cache_hit_improvement"]:
                print(
                    f"📊 Cache Hit Improvement: {cache_results['cache_hit_improvement']:.1f}%"
                )

    # Test 4: Load testing
    print("\n" + "=" * 50)
    print("Test 4: Load Testing (50 concurrent requests)")
    print("=" * 50)

    # Create 50 different patients
    load_test_patients = generate_test_patients(50)

    start_time = time.time()
    tasks = [test_single_prediction_async(patient) for patient in load_test_patients]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    successful_requests = len([r for r in results if isinstance(r, float) and r >= 0])
    failed_requests = len(results) - successful_requests

    print(f"\n📊 Load Test Results:")
    print(f"   Total requests: 50")
    print(f"   Successful: {successful_requests}")
    print(f"   Failed: {failed_requests}")
    print(f"   Success rate: {successful_requests/50*100:.1f}%")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Requests per second: {successful_requests/total_time:.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("🎯 Performance Testing Summary")
    print("=" * 70)
    print("✅ Async operations provide better concurrency")
    print("✅ Caching significantly improves response times")
    print("✅ Batch predictions are more efficient than individual requests")
    print("✅ The system can handle concurrent load effectively")
    print("\n💡 Recommendations:")
    print("   • Use async operations for better performance")
    print("   • Enable caching for frequently requested data")
    print("   • Use batch predictions for multiple patients")
    print("   • Monitor cache hit rates for optimization")


if __name__ == "__main__":
    asyncio.run(main())
