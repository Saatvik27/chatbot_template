#!/usr/bin/env python3
"""
Quick setup and dependency check for RAG Chatbot
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        'streamlit',
        'groq',
        'PyPDF2', 
        'sentence_transformers',
        'faiss_cpu',
        'python_dotenv',
        'pandas',
        'numpy',
        'langchain',
        'tiktoken'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    return len(missing) == 0

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data',
        'data/vector_store',
        'config'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def main():
    print("🚀 RAG Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed - could not install dependencies")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Some dependencies are missing")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Open your browser to http://localhost:8501")
    print("3. Go to Admin Panel and configure your Groq API key")
    print("4. Upload PDF documents")
    print("5. Start chatting!")
    print("\n📖 Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
