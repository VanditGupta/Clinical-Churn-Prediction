#!/usr/bin/env python3
"""
Comprehensive startup script for Clinical Study Churn Prediction services
Handles FastAPI backend, Streamlit frontend, and MLflow UI with proper port management
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

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Service configurations
SERVICES = {
    'fastapi': {
        'command': ['./venv/bin/uvicorn', 'api.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'],
        'url': 'http://localhost:8000/health',
        'name': 'FastAPI Backend',
        'port': 8000
    },
    'streamlit': {
        'command': ['./venv/bin/streamlit', 'run', 'app/dashboard.py', '--server.port', '8501', '--server.address', 'localhost'],
        'url': 'http://localhost:8501',
        'name': 'Streamlit Frontend',
        'port': 8501
    },
    'mlflow': {
        'command': ['./venv/bin/mlflow', 'ui', '--port', '8080', '--host', '0.0.0.0', '--workers', '1'],
        'url': 'http://localhost:8080',
        'name': 'MLflow UI',
        'port': 8080
    }
}

def check_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def wait_for_service(url, service_name, timeout=30):
    """Wait for a service to become available"""
    print(f"Waiting for {service_name} to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
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
    name = service_config['name']
    port = service_config['port']
    
    print(f"\n🚀 Starting {name} on port {port}...")
    
    # Check if port is available
    if not check_port_available(port):
        print(f"❌ Port {port} is already in use. Please stop the service using that port.")
        return None
    
    try:
        # Change to project root directory
        os.chdir(PROJECT_ROOT)
        
        # Start the service
        process = subprocess.Popen(
            service_config['command'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ {name} process started (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    """Main function to start all services"""
    print("=" * 70)
    print("🏥 Clinical Study Churn Prediction - Service Manager")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not (PROJECT_ROOT / 'api').exists():
        print("❌ Error: Please run this script from the project root directory")
        print(f"   Expected: {PROJECT_ROOT}")
        print(f"   Current: {Path.cwd()}")
        sys.exit(1)
    
    # Store running processes
    processes = {}
    
    try:
        # Start FastAPI backend first
        print("\n1️⃣ Starting FastAPI Backend...")
        fastapi_process = run_service(SERVICES['fastapi'], 'fastapi')
        if fastapi_process:
            processes['fastapi'] = fastapi_process
            # Wait for FastAPI to be ready
            if wait_for_service(SERVICES['fastapi']['url'], SERVICES['fastapi']['name']):
                print("   📊 API Documentation: http://localhost:8000/docs")
                print("   🔍 Alternative docs: http://localhost:8000/redoc")
            else:
                print("   ⚠️  FastAPI may not be fully ready yet")
        else:
            print("❌ Failed to start FastAPI backend")
            return
        
        # Start Streamlit frontend
        print("\n2️⃣ Starting Streamlit Frontend...")
        streamlit_process = run_service(SERVICES['streamlit'], 'streamlit')
        if streamlit_process:
            processes['streamlit'] = streamlit_process
            # Wait for Streamlit to be ready
            if wait_for_service(SERVICES['streamlit']['url'], SERVICES['streamlit']['name']):
                print("   🎯 Dashboard: http://localhost:8501")
            else:
                print("   ⚠️  Streamlit may not be fully ready yet")
        else:
            print("❌ Failed to start Streamlit frontend")
        
        # Start MLflow UI
        print("\n3️⃣ Starting MLflow UI...")
        mlflow_process = run_service(SERVICES['mlflow'], 'mlflow')
        if mlflow_process:
            processes['mlflow'] = mlflow_process
            # Wait for MLflow to be ready
            if wait_for_service(SERVICES['mlflow']['url'], SERVICES['mlflow']['name']):
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
        print("   🎯 Streamlit Dashboard: http://localhost:8501")
        print("   📈 MLflow UI:          http://localhost:8080")
        print("\n💡 Tips:")
        print("   • Use Ctrl+C to stop all services")
        print("   • Check the dashboard for interactive predictions")
        print("   • Use MLflow UI to explore model experiments")
        print("   • API endpoints are available for programmatic access")
        
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