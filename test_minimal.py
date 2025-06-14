import streamlit as st
import os
from datetime import datetime

# Minimal test app to check if basic Streamlit works
st.set_page_config(
    page_title="RAG Chatbot - Test Mode",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 RAG Chatbot - Test Mode")

st.markdown("""
## ✅ Basic Streamlit Test

This is a minimal version to test if Streamlit works without torch dependencies.

### 🔍 Dependency Check
""")

# Test basic imports
try:
    import groq
    st.success("✅ Groq - Available")
    
    # Test basic Groq functionality
    if st.button("Test Groq Import"):
        try:
            client = groq.Groq(api_key="test")  # This will fail but tests import
        except Exception as e:
            if "api_key" in str(e).lower():
                st.success("✅ Groq client can be initialized (API key needed)")
            else:
                st.error(f"❌ Groq error: {e}")
                
except ImportError:
    st.error("❌ Groq - Not installed")

try:
    import PyPDF2
    st.success("✅ PyPDF2 - Available")
except ImportError:
    st.error("❌ PyPDF2 - Not installed")

try:
    import faiss
    st.success("✅ FAISS - Available")
except ImportError:
    st.error("❌ FAISS - Not installed")

try:
    import numpy
    st.success("✅ NumPy - Available")
except ImportError:
    st.error("❌ NumPy - Not installed")

# Test sentence transformers separately
st.markdown("### 🤖 AI Model Test")
if st.button("Test Sentence Transformers"):
    try:
        from sentence_transformers import SentenceTransformer
        st.success("✅ Sentence Transformers - Import successful")
        
        with st.spinner("Loading model..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_text = ["Hello world", "Test sentence"]
            embeddings = model.encode(test_text)
            st.success(f"✅ Model loaded successfully! Embedding shape: {embeddings.shape}")
            
    except Exception as e:
        st.error(f"❌ Sentence Transformers error: {e}")
        st.info("This might be due to PyTorch compatibility issues with Python 3.13")

st.markdown("""
### 🚀 Next Steps

If all tests pass:
1. Run the main application: `streamlit run app.py`
2. Configure your Groq API key in Admin Panel
3. Upload PDF documents
4. Start chatting!

If Sentence Transformers fails:
- Try: `pip install --upgrade torch sentence-transformers`
- Or use Python 3.11 instead of 3.13
""")

st.markdown(f"**Current time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
