#!/usr/bin/env python3
"""
Comprehensive startup script for Clinical Study Churn Prediction services
Handles FastAPI backend, Streamlit frontend, MLflow UI, and Redis with proper port management
"""

import subprocess
import time
import sys
import os
import signal
import socket
from pathlib import Path
import threading
import requests
import asyncio

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Service configurations
SERVICES = {
    "redis": {
        "command": ["redis-server"],
        "url": None,  # Redis doesn't have HTTP endpoint
        "name": "Redis Cache",
        "port": 6379,
        "check_method": "redis_ping",
    },
    "fastapi": {
        "command": [
            "./venv/bin/uvicorn",
            "api.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "url": "http://localhost:8000/health",
        "name": "FastAPI Backend",
        "port": 8000,
        "check_method": "http",
    },
    "streamlit": {
        "command": [
            "./venv/bin/streamlit",
            "run",
            "app/dashboard.py",
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
        ],
        "url": "http://localhost:8501",
        "name": "Streamlit Frontend",
        "port": 8501,
        "check_method": "http",
    },
    "mlflow": {
        "command": [
            "./venv/bin/mlflow",
            "ui",
            "--port",
            "8080",
            "--host",
            "0.0.0.0",
            "--workers",
            "1",
        ],
        "url": "http://localhost:8080",
        "name": "MLflow UI",
        "port": 8080,
        "check_method": "http",
    },
}


def check_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def check_redis_ping():
    """Check if Redis is responding to ping"""
    try:
        result = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "PONG" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def wait_for_service(service_config, service_name, timeout=30):
    """Wait for a service to become available"""
    print(f"Waiting for {service_name} to start...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if service_config["check_method"] == "redis_ping":
            if check_redis_ping():
                print(f"✅ {service_name} is ready!")
                return True
        elif service_config["check_method"] == "http":
            try:
                response = requests.get(service_config["url"], timeout=2)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
        time.sleep(1)

    print(f"❌ {service_name} failed to start within {timeout} seconds")
    return False


def run_service(service_config, service_key):
    """Run a service in a subprocess"""
    name = service_config["name"]
    port = service_config["port"]

    print(f"\n🚀 Starting {name} on port {port}...")

    # Check if port is available (skip for Redis as it uses different protocol)
    if service_config["check_method"] != "redis_ping" and not check_port_available(
        port
    ):
        print(
            f"❌ Port {port} is already in use. Please stop the service using that port."
        )
        return None

    try:
        # Change to project root directory
        os.chdir(PROJECT_ROOT)

        # Start the service
        process = subprocess.Popen(
            service_config["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print(f"✅ {name} process started (PID: {process.pid})")
        return process

    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None


def setup_redis():
    """Setup Redis if not already running"""
    print("\n🔧 Setting up Redis...")

    # Check if Redis is installed
    try:
        result = subprocess.run(
            ["redis-server", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ Redis not found. Please install Redis first:")
            print("   python scripts/setup_redis.py")
            return False
    except FileNotFoundError:
        print("❌ Redis not found. Please install Redis first:")
        print("   python scripts/setup_redis.py")
        return False

    # Check if Redis is running
    if check_redis_ping():
        print("✅ Redis is already running")
        return True

    # Start Redis
    print("Starting Redis server...")
    try:
        redis_process = subprocess.Popen(
            ["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2)  # Wait for Redis to start

        if check_redis_ping():
            print("✅ Redis started successfully!")
            return True
        else:
            print("❌ Redis failed to start")
            return False
    except Exception as e:
        print(f"❌ Failed to start Redis: {e}")
        return False


def main():
    """Main function to start all services"""
    print("=" * 70)
    print("🏥 Clinical Study Churn Prediction - Enhanced Service Manager")
    print("=" * 70)

    # Check if we're in the right directory
    if not (PROJECT_ROOT / "api").exists():
        print("❌ Error: Please run this script from the project root directory")
        print(f"   Expected: {PROJECT_ROOT}")
        print(f"   Current: {Path.cwd()}")
        sys.exit(1)

    # Store running processes
    processes = {}

    try:
        # Setup Redis first
        print("\n1️⃣ Setting up Redis Cache...")
        if not setup_redis():
            print(
                "⚠️  Redis setup failed. Continuing without Redis (using in-memory cache only)"
            )
        else:
            print("   📊 Redis will be used for caching")

        # Start FastAPI backend
        print("\n2️⃣ Starting FastAPI Backend...")
        fastapi_process = run_service(SERVICES["fastapi"], "fastapi")
        if fastapi_process:
            processes["fastapi"] = fastapi_process
            # Wait for FastAPI to be ready
            if wait_for_service(SERVICES["fastapi"], SERVICES["fastapi"]["name"]):
                print("   📊 API Documentation: http://localhost:8000/docs")
                print("   🔍 Alternative docs: http://localhost:8000/redoc")
                print("   📈 Cache stats: http://localhost:8000/cache/stats")
            else:
                print("   ⚠️  FastAPI may not be fully ready yet")
        else:
            print("❌ Failed to start FastAPI backend")
            return

        # Start Streamlit frontend
        print("\n3️⃣ Starting Streamlit Frontend...")
        streamlit_process = run_service(SERVICES["streamlit"], "streamlit")
        if streamlit_process:
            processes["streamlit"] = streamlit_process
            # Wait for Streamlit to be ready
            if wait_for_service(SERVICES["streamlit"], SERVICES["streamlit"]["name"]):
                print("   🎯 Dashboard: http://localhost:8501")
            else:
                print("   ⚠️  Streamlit may not be fully ready yet")
        else:
            print("❌ Failed to start Streamlit frontend")

        # Start MLflow UI
        print("\n4️⃣ Starting MLflow UI...")
        mlflow_process = run_service(SERVICES["mlflow"], "mlflow")
        if mlflow_process:
            processes["mlflow"] = mlflow_process
            # Wait for MLflow to be ready
            if wait_for_service(SERVICES["mlflow"], SERVICES["mlflow"]["name"]):
                print("   📈 MLflow UI: http://localhost:8080")
            else:
                print("   ⚠️  MLflow may not be fully ready yet")
        else:
            print("❌ Failed to start MLflow UI")

        # Summary
        print("\n" + "=" * 70)
        print("🎉 All services started successfully!")
        print("=" * 70)
        print("\n📋 Service URLs:")
        print("   🏥 FastAPI Backend:    http://localhost:8000")
        print("   📊 API Documentation:  http://localhost:8000/docs")
        print("   📈 Cache Statistics:   http://localhost:8000/cache/stats")
        print("   🎯 Streamlit Dashboard: http://localhost:8501")
        print("   📈 MLflow UI:          http://localhost:8080")
        print("\n🚀 New Features:")
        print("   • Async operations for better performance")
        print("   • Redis caching for faster responses")
        print("   • Batch predictions for multiple patients")
        print("   • Background task processing")
        print("   • Enhanced error handling and monitoring")
        print("\n💡 Tips:")
        print("   • Use Ctrl+C to stop all services")
        print("   • Check cache statistics for performance insights")
        print("   • Try batch predictions for multiple patients")
        print("   • Use async operations for better responsiveness")

        # Keep the script running
        print("\n⏳ Services are running... Press Ctrl+C to stop all services")

        # Wait for interrupt
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")

        # Stop all processes
        for service_name, process in processes.items():
            if process and process.poll() is None:
                print(f"   Stopping {SERVICES[service_name]['name']}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print(f"   ✅ {SERVICES[service_name]['name']} stopped")

        print("\n👋 All services stopped. Goodbye!")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

        # Clean up processes
        for process in processes.values():
            if process and process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
