# Set up environment variables first
import env_setup

# Apply patches before importing anything else to fix PyTorch/Streamlit compatibility
import streamlit_patch

import streamlit as st
import os
import sys

# Add error handling for PyTorch compatibility issues
def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    try:
        import groq
    except ImportError:
        missing_deps.append("groq")
    
    try:
        import PyPDF2
    except ImportError:
        missing_deps.append("PyPDF2")
    
    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # Don't check sentence_transformers here to avoid torch issues
    
    return missing_deps

# Check dependencies at startup
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing dependencies: {', '.join(missing_deps)}")
    st.info("Please run: pip install -r requirements.txt")
    st.stop()

from groq import Groq
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

# Import our custom modules
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.chat_manager import ChatManager
from config.settings import load_config, save_config

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: #333333;
    }
    .bot-message {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        color: #333333;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: none;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Load configuration
    config = load_config()
    
    # Sidebar navigation
    st.sidebar.title("🤖 RAG Chatbot")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["💬 Chat", "👨‍💼 Admin Panel", "📊 Analytics", "ℹ️ About"]
    )
    
    # Navigation logic
    if page == "💬 Chat":
        show_chat_page(config)
    elif page == "👨‍💼 Admin Panel":
        show_admin_page(config)
    elif page == "📊 Analytics":
        show_analytics_page()
    else:
        show_about_page()

def show_chat_page(config):
    st.markdown('<h1 class="main-header">💬 Chat with AI Assistant</h1>', unsafe_allow_html=True)
    
    # Check if Groq API key is configured
    if not config.get('groq_api_key'):
        st.error("⚠️ Groq API key not configured. Please contact admin to set up the API key.")
        return
    
    # Check if vector store is available
    vector_store = VectorStore()
    has_documents = vector_store.is_initialized()
    
    # Display document status
    if has_documents:
        info = vector_store.get_info()
        st.success(f"📚 Knowledge base loaded with {info['total_documents']} documents")
        st.info("� Ask questions about your documents or have a general conversation!")
    else:
        st.info("💭 No documents uploaded yet. You can still have a general conversation with the AI!")
        st.markdown("**Note:** Responses will be based on the AI's general knowledge. Upload documents in Admin Panel for document-specific answers.")
    
    # Chat interface
    st.markdown("### Ask me anything!")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.chat_manager.get_response(
                    prompt, 
                    config['groq_api_key'],
                    config.get('model_settings', {}).get('groq_model', 'llama-3.3-70b-versatile')
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {response}</div>', unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def show_admin_page(config):
    st.markdown('<h1 class="main-header">👨‍💼 Admin Panel</h1>', unsafe_allow_html=True)
    
    # Admin authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.markdown("### 🔐 Admin Login")
        password = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if password == config.get('admin_password', 'admin123'):  # Default password
                st.session_state.admin_authenticated = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password!")
        st.info("💡 Default password is 'admin123'. You can change it in the configuration section.")
        return
    
    # Admin dashboard
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔑 API Configuration")
        
        # Groq API Key input
        groq_key = st.text_input(
            "Groq API Key:",
            value=config.get('groq_api_key', ''),
            type="password",
            help="Enter your Groq API key for Llama model access"
        )
        
        if st.button("💾 Save API Key"):
            config['groq_api_key'] = groq_key
            save_config(config)
            st.success("✅ API key saved successfully!")
        
        # Test API key
        if groq_key and st.button("🧪 Test API Key"):
            try:
                client = Groq(api_key=groq_key)
                # Test with a simple completion
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="llama-3.3-70b-versatile",
                    max_tokens=10
                )
                st.success("✅ API key is valid!")
            except Exception as e:
                st.error(f"❌ API key test failed: {str(e)}")
        
        # Admin password change
        st.markdown("### 🔒 Change Admin Password")
        new_password = st.text_input("New Password:", type="password")
        if st.button("🔄 Update Password") and new_password:
            config['admin_password'] = new_password
            save_config(config)
            st.success("✅ Password updated successfully!")
    
    with col2:
        st.markdown("### 📄 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents:",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to build the knowledge base"
        )
        
        if uploaded_files and st.button("🚀 Process Documents"):
            pdf_processor = PDFProcessor()
            vector_store = VectorStore()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                all_texts = []
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text from PDF
                    texts = pdf_processor.extract_text_from_pdf(temp_path)
                    all_texts.extend(texts)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Create vector store
                status_text.text("Creating vector embeddings...")
                vector_store.create_vector_store(all_texts)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Documents processed successfully!")
                st.success(f"📚 Processed {len(uploaded_files)} documents with {len(all_texts)} text chunks!")
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
        
        # Display current documents info
        vector_store = VectorStore()
        if vector_store.is_initialized():
            st.markdown("### 📊 Current Knowledge Base")
            info = vector_store.get_info()
            st.info(f"📈 Total documents: {info['total_documents']}")
            
            if st.button("🗑️ Clear Knowledge Base"):
                vector_store.clear()
                st.success("✅ Knowledge base cleared!")
                st.rerun()
        
        # Model selection
        st.markdown("### 🤖 Model Configuration")
          # Available models (updated with currently supported models)
        available_models = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
            "llama-3.1-8b-instant": "Llama 3.1 8B (Faster)",
            "gemma2-9b-it": "Gemma 2 9B",
            "llama-3.3-8b-instant": "Llama 3.3 8B (Fast)",
            "llama-3.1-70b-versatile": "Llama 3.1 70B (Alternative)"
        }
        
        current_model = config.get('model_settings', {}).get('groq_model', 'llama-3.3-70b-versatile')
        
        selected_model = st.selectbox(
            "Select Groq Model:",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            index=list(available_models.keys()).index(current_model) if current_model in available_models else 0,
            help="Choose the AI model for chat responses"
        )
        
        if st.button("💾 Save Model Selection"):
            config['model_settings']['groq_model'] = selected_model
            save_config(config)
            st.success(f"✅ Model updated to: {available_models[selected_model]}")
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

def show_analytics_page():
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load chat history for analytics
    chat_history_file = "data/chat_history.json"
    
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        if chat_data:
            # Convert to DataFrame
            df = pd.DataFrame(chat_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Conversations", len(df))
                st.metric("Unique Users", df['user_id'].nunique() if 'user_id' in df.columns else "N/A")
            
            with col2:
                if len(df) > 0:
                    today = datetime.now().date()
                    today_chats = df[df['timestamp'].dt.date == today]
                    st.metric("Today's Chats", len(today_chats))
                    
                    # Calculate average response time, handling missing values
                    if 'response_time' in df.columns and df['response_time'].notna().any():
                        avg_response_time = df['response_time'].mean()
                        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
                    else:
                        st.metric("Avg Response Time", "N/A")
              # Chat frequency over time
            st.markdown("### 📈 Chat Activity Over Time")
            try:
                # Group by date and count chats
                daily_chats = df.groupby(df['timestamp'].dt.date).size().reset_index()
                daily_chats.columns = ['Date', 'Number of Chats']
                
                if len(daily_chats) > 0:
                    # Use Altair chart for better control
                    import altair as alt
                    
                    # Create interactive selection (fix deprecation warning)
                    selection = alt.selection_interval()
                    
                    chart = alt.Chart(daily_chats).mark_line(
                        point=True,
                        strokeWidth=4,
                        color='#1f77b4'
                    ).add_params(
                        selection
                    ).encode(
                        x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('Number of Chats:Q', title='Number of Chats'),
                        tooltip=['Date:T', 'Number of Chats:Q']
                    ).properties(
                        width='container',
                        height=400,
                        title=alt.TitleParams(
                            text='Daily Chat Activity',
                            fontSize=16,
                            anchor='start'
                        )
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No chat activity data available yet.")
                    
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                # Fallback: simple metrics
                daily_chats = df.groupby(df['timestamp'].dt.date).size()
                if len(daily_chats) > 0:
                    st.write("**Daily Chat Counts:**")
                    for date, count in daily_chats.items():
                        st.write(f"• {date}: {count} chats")
            
            # Hourly activity analysis
            st.markdown("### 🕐 Chat Activity by Hour")
            try:
                hourly_chats = df.groupby(df['timestamp'].dt.hour).size().reset_index()
                hourly_chats.columns = ['Hour', 'Number of Chats']
                
                if len(hourly_chats) > 0:
                    import altair as alt
                    
                    hour_chart = alt.Chart(hourly_chats).mark_bar(
                        color='#2ca02c',
                        cornerRadiusTopLeft=3,
                        cornerRadiusTopRight=3
                    ).encode(
                        x=alt.X('Hour:O', title='Hour of Day', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Number of Chats:Q', title='Number of Chats'),
                        tooltip=['Hour:O', 'Number of Chats:Q']
                    ).properties(
                        width='container',
                        height=350,
                        title=alt.TitleParams(
                            text='Chat Activity by Hour of Day',
                            fontSize=16,
                            anchor='start'
                        )
                    )
                    
                    st.altair_chart(hour_chart, use_container_width=True)
                else:
                    st.info("No hourly activity data available yet.")
                    
            except Exception as e:
                st.error(f"Error creating hourly chart: {str(e)}")
            
            # Recent conversations
            st.markdown("### 💬 Recent Conversations")
            recent_df = df.tail(10)[['timestamp', 'question', 'response']].copy()
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("📭 No chat data available yet.")
    else:
        st.info("📭 No analytics data available yet. Start chatting to see analytics!")

def show_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About RAG Chatbot</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Features
    
    - **🤖 Intelligent Chat**: Powered by Llama 3.1 70B via Groq API
    - **📚 Document Processing**: Upload and process PDF documents
    - **🔍 Semantic Search**: Uses sentence transformers for relevant context retrieval
    - **👨‍💼 Admin Panel**: Easy document and API key management
    - **📊 Analytics**: Track usage and performance metrics
    - **🎨 Beautiful UI**: Clean and responsive interface
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **LLM**: Llama 3.1 70B via Groq
    - **Embeddings**: Sentence Transformers
    - **Vector Store**: FAISS
    - **PDF Processing**: PyPDF2
    - **Backend**: Python
    
    ### 🔧 Setup Instructions
    
    1. **Install Dependencies**: `pip install -r requirements.txt`
    2. **Run the App**: `streamlit run app.py`
    3. **Admin Setup**: Go to Admin Panel and configure Groq API key
    4. **Upload Documents**: Add PDF files to build knowledge base
    5. **Start Chatting**: Use the chat interface to ask questions
    
    ### 📝 Usage Tips
    
    - Upload relevant PDF documents for better responses
    - Ask specific questions for more accurate answers
    - Use the admin panel to manage the knowledge base
    - Check analytics to monitor usage patterns
    
    ### 🔐 Security
    
    - Admin authentication required for sensitive operations
    - API keys are stored securely
    - Chat history is logged for analytics
    
    ### 📞 Support
    
    For technical support or questions, please refer to the documentation or contact the system administrator.
    """)

if __name__ == "__main__":
    main()
