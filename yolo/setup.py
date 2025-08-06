"""
Setup script for YOLO Weight Display Reader
"""

import subprocess
import sys
import os
import requests
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def download_yolo_model():
    """Download YOLOv8 model if not present"""
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv8n model...")
        try:
            # The ultralytics package will automatically download the model
            # when first used, so we don't need to manually download
            print("✓ YOLOv8 model will be downloaded automatically on first use")
            return True
        except Exception as e:
            print(f"✗ Failed to setup YOLOv8 model: {e}")
            return False
    else:
        print("✓ YOLOv8 model already exists")
        return True

def main():
    print("YOLO Weight Display Reader Setup")
    print("=" * 50)
    
    # Install required packages
    print("Installing required packages...")
    packages = [
        "ultralytics",
        "opencv-python", 
        "torch",
        "torchvision",
        "easyocr",
        "numpy",
        "Pillow",
        "matplotlib",
        "requests"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nPackage installation: {success_count}/{len(packages)} successful")
    
    # Download YOLO model
    print("\nSetting up YOLO model...")
    download_yolo_model()
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nUsage examples:")
    print("1. Process an image:")
    print("   python yolo.py --input image.jpg --output result.jpg")
    print("\n2. Process a video:")
    print("   python yolo.py --input video.mp4 --output result.mp4 --video")
    print("\n3. Use GPU acceleration:")
    print("   python yolo.py --input image.jpg --gpu")

if __name__ == "__main__":
    main()
