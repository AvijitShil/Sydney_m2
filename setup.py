#!/usr/bin/env python3
"""
Medical AI Assistant Setup Script
Handles installation, configuration, and first-time setup.
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_system_requirements():
    """Check system requirements and dependencies."""
    system = platform.system()
    print(f"🖥️ Operating System: {system}")
    
    # Check for required system packages
    if system == "Windows":
        print("📋 Windows detected - ensure you have:")
        print("   • Microsoft Visual C++ Redistributable")
        print("   • Windows Media Feature Pack (for audio)")
    elif system == "Linux":
        print("📋 Linux detected - ensure you have:")
        print("   • sudo apt-get install portaudio19-dev python3-pyaudio")
        print("   • sudo apt-get install ffmpeg")
    elif system == "Darwin":  # macOS
        print("📋 macOS detected - ensure you have:")
        print("   • brew install portaudio")
        print("   • brew install ffmpeg")

def install_ollama():
    """Install Ollama if not present."""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("✅ Ollama is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("🔄 Installing Ollama...")
        system = platform.system()
        
        if system == "Windows":
            print("📥 Please download and install Ollama from: https://ollama.ai/download/windows")
            input("Press Enter after installing Ollama...")
        elif system == "Linux":
            subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], shell=True)
        elif system == "Darwin":
            print("📥 Please download and install Ollama from: https://ollama.ai/download/mac")
            input("Press Enter after installing Ollama...")

def pull_ollama_model(model_name: str = "gemma:1b"):
    """Pull the required Ollama model."""
    try:
        print(f"🔄 Pulling Ollama model: {model_name}")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Successfully pulled {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull {model_name}: {e}")
        print("🔧 You can manually run: ollama pull gemma:1b")

def install_requirements():
    """Install Python requirements."""
    try:
        print("🔄 Installing Python requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment variables and configuration."""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("🔧 Creating environment configuration...")
        with open(env_file, "w") as f:
            f.write("ENVIRONMENT=development\n")
            f.write("CUDA_VISIBLE_DEVICES=0\n")
            f.write("TOKENIZERS_PARALLELISM=false\n")
        print("✅ Environment file created")

def test_audio_system():
    """Test audio input/output capabilities."""
    try:
        import sounddevice as sd
        print("🔊 Testing audio system...")
        
        # List audio devices
        devices = sd.query_devices()
        print(f"📱 Found {len(devices)} audio devices")
        
        # Test recording
        print("🎤 Testing microphone (2 second test)...")
        duration = 2
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Simple volume check
        volume = float(np.sqrt(np.mean(audio_data**2)))
        if volume > 0.001:
            print("✅ Microphone working")
        else:
            print("⚠️ Microphone might not be working properly")
            
    except Exception as e:
        print(f"⚠️ Audio system test failed: {e}")
        print("📋 Please check your audio drivers and microphone permissions")

def create_directories():
    """Create necessary directories."""
    directories = [
        "./medical_knowledge_db",
        "./logs",
        "./cache",
        "./models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def display_hardware_info():
    """Display hardware information and recommendations."""
    try:
        import torch
        import psutil
        
        print("\n🖥️ Hardware Information:")
        print(f"   CPU Cores: {psutil.cpu_count(logical=False)}")
        print(f"   CPU Threads: {psutil.cpu_count(logical=True)}")
        print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            print("✅ CUDA acceleration available")
        else:
            print("⚠️ No CUDA GPU detected - will use CPU")
            
    except ImportError:
        print("⚠️ Could not check hardware info (torch not installed yet)")

def main():
    """Main setup function."""
    print("🏥 Medical AI Assistant Setup")
    print("=" * 40)
    
    # Basic checks
    check_python_version()
    display_hardware_info()
    check_system_requirements()
    
    # Environment setup
    setup_environment()
    create_directories()
    
    # Install dependencies
    install_ollama()
    install_requirements()
    
    # Pull required models
    pull_ollama_model("gemma:1b")
    
    # Test systems
    test_audio_system()
    
    print("\n✅ Setup complete!")
    print("\n🚀 To run the Medical AI Assistant:")
    print("   • Basic version: python main.py")
    print("   • Optimized version: python main_optimized.py")
    print("\n⚕️ Remember: This is for educational purposes only.")
    print("   Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()
