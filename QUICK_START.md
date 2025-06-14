# 🚀 RAG Chatbot - Quick Start Guide

## 📋 What You Need

1. **Python 3.8+** installed on your system
2. **Groq API Key** (free from [console.groq.com](https://console.groq.com))
3. **PDF documents** to create a knowledge base

## ⚡ Quick Setup (5 minutes)

### Option 1: Automatic Setup (Recommended)
```bash
# Run the setup script
python setup.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 3: Windows Batch Files
```bash
# Install dependencies
install.bat

# Run the application
run.bat
```

## 🔧 Configuration Steps

### 1. Get Your Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Generate an API key (starts with `gsk_`)
4. Copy the key for the next step

### 2. Configure the Application
1. Open the application in your browser (http://localhost:8501)
2. Go to "👨‍💼 Admin Panel" in the sidebar
3. Login with password: `admin123`
4. Paste your Groq API key in the "API Configuration" section
5. Click "💾 Save API Key"
6. Click "🧪 Test API Key" to verify it works

### 3. Upload Documents
1. In the Admin Panel, go to "📄 Document Management"
2. Click "Choose files" and select your PDF documents
3. Click "🚀 Process Documents"
4. Wait for processing to complete

### 4. Start Chatting
1. Go to "💬 Chat" in the sidebar
2. Type your questions about the uploaded documents
3. Get intelligent answers based on your documents!

## 📊 Features Overview

### 💬 Chat Interface
- Ask questions about your documents
- Get contextual answers from Mixtral AI
- View chat history
- Clear chat when needed

### 👨‍💼 Admin Panel
- **API Configuration**: Manage Groq API key
- **Document Management**: Upload and process PDFs
- **Password Management**: Change admin password
- **Knowledge Base**: View and manage documents

### 📊 Analytics
- View chat statistics
- Monitor usage patterns
- Track response times
- Analyze user interactions

### ℹ️ About Page
- Feature overview
- Technology stack
- Setup instructions
- Usage tips

## 🎯 Usage Tips

### For Best Results:
1. **Upload relevant documents** - The bot can only answer based on uploaded content
2. **Ask specific questions** - More specific questions get better answers
3. **Use natural language** - Ask questions as you would to a human
4. **Check context** - The bot shows relevance scores for retrieved content

### Example Questions:
- "What are the main benefits mentioned in the document?"
- "How do I configure the system according to the manual?"
- "What are the requirements for this project?"
- "Summarize the key points from chapter 3"

## 🔧 Customization

### Change Models
Edit `config/app_config.json`:
```json
{
  "model_settings": {
    "embedding_model": "all-MiniLM-L6-v2",
    "groq_model": "mixtral-8x7b-32768",
    "temperature": 0.1,
    "chunk_size": 1000
  }
}
```

### Modify UI Colors
Edit the CSS in `app.py`:
```python
st.markdown("""
<style>
    .main-header {
        color: #YOUR_COLOR;
    }
</style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### Common Issues:

**"Groq API key not configured"**
- Go to Admin Panel and add your API key
- Make sure the key starts with `gsk_`

**"No documents uploaded"**
- Upload PDF files in the Admin Panel
- Wait for processing to complete

**"Import errors"**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**"Can't access admin panel"**
- Default password is `admin123`
- Check if you're on the correct page

### Getting Help:
1. Check the terminal for error messages
2. Verify all dependencies are installed: `python setup.py`
3. Test individual components: `streamlit run test_setup.py`

## 📚 File Structure

```
chatbot_template/
├── app.py              # Main application
├── setup.py            # Setup script
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
├── config/
│   └── settings.py    # Configuration management
├── utils/
│   ├── pdf_processor.py   # PDF processing
│   ├── vector_store.py    # Vector database
│   └── chat_manager.py    # Chat logic
└── data/
    ├── vector_store/      # Vector database files
    └── chat_history.json  # Chat logs
```

## 🔐 Security Notes

- Change the default admin password (`admin123`)
- Keep your Groq API key secure
- The app stores data locally in the `data/` folder
- No user data is sent to external services except Groq

## 🚀 Next Steps

1. **Add more documents** - Upload more PDFs to expand knowledge
2. **Customize the UI** - Modify colors, layout, and branding
3. **Add features** - Implement user authentication, API endpoints
4. **Scale up** - Deploy to cloud platforms like Streamlit Cloud

## 📞 Support

- Check the main README.md for detailed documentation
- Run `streamlit run test_setup.py` to test your setup
- Look at the terminal output for error messages

---

**Happy chatting with your RAG bot! 🤖✨**
