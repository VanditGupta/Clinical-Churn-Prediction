#!/usr/bin/env python3
"""
Port availability checker for Clinical Study Churn Prediction services
Checks if required ports are available before starting services
"""

import socket
import sys
from pathlib import Path

# Service ports
PORTS = {
    8000: "FastAPI Backend",
    8501: "Streamlit Frontend", 
    8080: "MLflow UI"
}

def check_port(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def get_process_using_port(port):
    """Get process information for a port (macOS/Linux)"""
    try:
        import subprocess
        result = subprocess.run(
            ['lsof', '-i', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header line
                process_info = lines[1].split()
                if len(process_info) >= 2:
                    return f"PID {process_info[1]} ({process_info[0]})"
        return "Unknown process"
    except Exception:
        return "Unable to determine"

def main():
    """Main function to check all ports"""
    print("🔍 Port Availability Check for Clinical Study Churn Prediction")
    print("=" * 60)
    
    all_available = True
    
    for port, service in PORTS.items():
        print(f"\n📡 Checking {service} (port {port})...")
        
        if check_port(port):
            print(f"   ✅ Port {port} is available")
        else:
            print(f"   ❌ Port {port} is in use")
            process_info = get_process_using_port(port)
            print(f"      Process: {process_info}")
            all_available = False
    
    print("\n" + "=" * 60)
    
    if all_available:
        print("🎉 All ports are available! You can start the services.")
        print("\nTo start all services:")
        print("   python scripts/start_services.py")
    else:
        print("⚠️  Some ports are in use. Please stop the conflicting processes.")
        print("\nTo stop processes using these ports:")
        print("   # For macOS/Linux:")
        print("   lsof -ti:8000 | xargs kill -9  # Stop FastAPI")
        print("   lsof -ti:8501 | xargs kill -9  # Stop Streamlit")
        print("   lsof -ti:8080 | xargs kill -9  # Stop MLflow")
        print("\n   # Or use the process manager:")
        print("   python scripts/start_services.py  # Will handle conflicts")
    
    print("\nService URLs (when running):")
    print("   🏥 FastAPI Backend:    http://localhost:8000")
    print("   📊 API Documentation:  http://localhost:8000/docs")
    print("   🎯 Streamlit Dashboard: http://localhost:8501")
    print("   📈 MLflow UI:          http://localhost:8080")

if __name__ == "__main__":
    main() 