import streamlit as st
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    st.set_page_config(
        page_title="RAG Chatbot Demo",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 RAG Chatbot - Quick Test")
    
    st.markdown("""
    This is a quick test page to verify that all components are working correctly.
    
    ## ✅ Component Status Check
    """)
    
    # Test imports
    try:
        from utils.pdf_processor import PDFProcessor
        st.success("✅ PDF Processor - OK")
    except ImportError as e:
        st.error(f"❌ PDF Processor - Error: {e}")
    
    try:
        from utils.vector_store import VectorStore
        st.success("✅ Vector Store - OK")
    except ImportError as e:
        st.error(f"❌ Vector Store - Error: {e}")
    
    try:
        from utils.chat_manager import ChatManager
        st.success("✅ Chat Manager - OK")
    except ImportError as e:
        st.error(f"❌ Chat Manager - Error: {e}")
    
    try:
        from config.settings import load_config
        config = load_config()
        st.success("✅ Configuration - OK")
        st.json(config)
    except ImportError as e:
        st.error(f"❌ Configuration - Error: {e}")
    
    # Test dependencies
    st.markdown("## 📦 Dependencies Check")
    
    dependencies = [
        'streamlit', 'PyPDF2', 'sentence_transformers', 
        'faiss_cpu', 'numpy', 'pandas', 'groq'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            st.success(f"✅ {dep} - Installed")
        except ImportError:
            st.error(f"❌ {dep} - Not installed")
    
    st.markdown("""
    ## 🚀 Next Steps
    
    If all components show "OK", you can:
    1. Run the main application: `streamlit run app.py`
    2. Go to Admin Panel and configure your Groq API key
    3. Upload PDF documents
    4. Start chatting!
    
    ## 📖 Documentation
    
    Check the README.md file for detailed setup instructions.
    """)

if __name__ == "__main__":
    main()
