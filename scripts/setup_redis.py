#!/usr/bin/env python3
"""
Redis Setup Script for Clinical Study Churn Prediction
Helps users set up Redis for caching functionality
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def check_redis_installed():
    """Check if Redis is installed"""
    try:
        result = subprocess.run(
            ["redis-server", "--version"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_redis_running():
    """Check if Redis is running"""
    try:
        result = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "PONG" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_redis_macos():
    """Install Redis on macOS using Homebrew"""
    print("Installing Redis on macOS...")
    try:
        # Check if Homebrew is installed
        result = subprocess.run(["brew", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print(
                '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            )
            return False

        # Install Redis
        subprocess.run(["brew", "install", "redis"], check=True)
        print("✅ Redis installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Redis: {e}")
        return False


def install_redis_ubuntu():
    """Install Redis on Ubuntu/Debian"""
    print("Installing Redis on Ubuntu/Debian...")
    try:
        # Update package list
        subprocess.run(["sudo", "apt", "update"], check=True)

        # Install Redis
        subprocess.run(["sudo", "apt", "install", "-y", "redis-server"], check=True)

        # Start Redis service
        subprocess.run(["sudo", "systemctl", "start", "redis-server"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "redis-server"], check=True)

        print("✅ Redis installed and started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Redis: {e}")
        return False


def install_redis_centos():
    """Install Redis on CentOS/RHEL"""
    print("Installing Redis on CentOS/RHEL...")
    try:
        # Install EPEL repository
        subprocess.run(["sudo", "yum", "install", "-y", "epel-release"], check=True)

        # Install Redis
        subprocess.run(["sudo", "yum", "install", "-y", "redis"], check=True)

        # Start Redis service
        subprocess.run(["sudo", "systemctl", "start", "redis"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "redis"], check=True)

        print("✅ Redis installed and started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Redis: {e}")
        return False


def start_redis():
    """Start Redis server"""
    print("Starting Redis server...")
    try:
        # Start Redis in background
        subprocess.Popen(
            ["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait a moment for Redis to start
        import time

        time.sleep(2)

        # Test connection
        if check_redis_running():
            print("✅ Redis server started successfully!")
            return True
        else:
            print("❌ Redis server failed to start")
            return False
    except Exception as e:
        print(f"❌ Failed to start Redis: {e}")
        return False


def test_redis_connection():
    """Test Redis connection"""
    print("Testing Redis connection...")
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful!")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("Redis Setup for Clinical Study Churn Prediction")
    print("=" * 60)

    # Check if Redis is already installed
    if check_redis_installed():
        print("✅ Redis is already installed")
    else:
        print("❌ Redis is not installed")

        # Detect operating system
        system = platform.system().lower()

        if system == "darwin":  # macOS
            if not install_redis_macos():
                print("\n❌ Failed to install Redis. Please install manually:")
                print("   brew install redis")
                return False
        elif system == "linux":
            # Try to detect Linux distribution
            try:
                with open("/etc/os-release", "r") as f:
                    content = f.read().lower()
                    if "ubuntu" in content or "debian" in content:
                        if not install_redis_ubuntu():
                            print(
                                "\n❌ Failed to install Redis. Please install manually:"
                            )
                            print("   sudo apt install redis-server")
                            return False
                    elif "centos" in content or "rhel" in content:
                        if not install_redis_centos():
                            print(
                                "\n❌ Failed to install Redis. Please install manually:"
                            )
                            print("   sudo yum install redis")
                            return False
                    else:
                        print(
                            "❌ Unsupported Linux distribution. Please install Redis manually."
                        )
                        return False
            except FileNotFoundError:
                print(
                    "❌ Could not detect Linux distribution. Please install Redis manually."
                )
                return False
        else:
            print(f"❌ Unsupported operating system: {system}")
            print("Please install Redis manually for your system.")
            return False

    # Check if Redis is running
    if check_redis_running():
        print("✅ Redis is already running")
    else:
        print("❌ Redis is not running")
        if not start_redis():
            print("\n❌ Failed to start Redis. Please start manually:")
            print("   redis-server")
            return False

    # Test Python Redis connection
    if test_redis_connection():
        print("\n🎉 Redis setup completed successfully!")
        print("\nYou can now run the application with caching enabled.")
        return True
    else:
        print("\n❌ Redis setup failed. Please check the installation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
