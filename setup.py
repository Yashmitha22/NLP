"""
Setup script for Voice Assistant
This script helps install the required dependencies for the voice assistant.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    print("Voice Assistant Setup")
    print("=" * 40)
    
    # List of required packages
    packages = [
        "speechrecognition",
        "pyttsx3", 
        "wikipedia",
        "requests"
    ]
    
    print("Installing required packages...")
    print()
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully.")
    
    # Special handling for PyAudio (can be tricky on Windows)
    print("\nAttempting to install PyAudio...")
    if not install_package("pyaudio"):
        print("\n⚠️  PyAudio installation failed. This is common on Windows.")
        print("You may need to:")
        print("1. Install Microsoft C++ Build Tools")
        print("2. Or use: pip install pipwin && pipwin install pyaudio")
        print("3. Or download a wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("You can now run the voice assistant with: python assistant.py")

if __name__ == "__main__":
    main()
