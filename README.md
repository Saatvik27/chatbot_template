# RAG Chatbot Template

A complete Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, Groq (Mixtral), and Sentence Transformers.

## 🚀 Features

- **🤖 Intelligent Chat**: Powered by Mixtral-8x7B via Groq API
- **📚 Document Processing**: Upload and process PDF documents
- **🔍 Semantic Search**: Uses sentence transformers for relevant context retrieval
- **👨‍💼 Admin Panel**: Easy document and API key management
- **📊 Analytics**: Track usage and performance metrics
- **🎨 Beautiful UI**: Clean and responsive interface

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [Groq Console](https://console.groq.com/))

## 🛠️ Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd chatbot_template
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## 🔧 Setup Instructions

### 1. Initial Setup
1. Run the application using `streamlit run app.py`
2. Navigate to the "👨‍💼 Admin Panel" in the sidebar
3. Login with the default password: `admin123`

### 2. Configure Groq API Key
1. In the Admin Panel, enter your Groq API key
2. Click "💾 Save API Key"
3. Test the API key by clicking "🧪 Test API Key"

### 3. Upload Documents
1. In the Admin Panel, go to "📄 Document Management"
2. Upload PDF files using the file uploader
3. Click "🚀 Process Documents" to create vector embeddings

### 4. Start Chatting
1. Go to "💬 Chat" in the sidebar
2. Ask questions about your uploaded documents
3. The bot will provide answers based on the document content

## 📁 Project Structure

```
chatbot_template/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration management
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py  # PDF text extraction
│   ├── vector_store.py   # Vector database management
│   └── chat_manager.py   # Chat logic and RAG implementation
└── data/
    ├── README.md
    ├── vector_store/     # FAISS vector database (auto-created)
    └── chat_history.json # Chat logs (auto-created)
```

## 🔑 Configuration

### Environment Variables
You can also set configuration using environment variables:
- `GROQ_API_KEY`: Your Groq API key
- `ADMIN_PASSWORD`: Admin panel password

### Configuration File
The app creates a configuration file at `config/app_config.json` with the following structure:

```json
{
    "groq_api_key": "your-groq-api-key",
    "admin_password": "admin123",
    "model_settings": {
        "embedding_model": "all-MiniLM-L6-v2",
        "groq_model": "mixtral-8x7b-32768",
        "max_tokens": 1024,
        "temperature": 0.1,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}
```

## 📖 Usage

### For End Users
1. **Ask Questions**: Type your questions in the chat interface
2. **Get Contextual Answers**: The bot will search through uploaded documents and provide relevant answers
3. **View Chat History**: Scroll up to see previous conversations

### For Administrators
1. **Upload Documents**: Add PDF files to expand the knowledge base
2. **Manage API Keys**: Update Groq API key as needed
3. **Monitor Usage**: Check analytics for usage patterns
4. **Clear Data**: Reset chat history or knowledge base when needed

## 🎯 Key Components

### 1. PDF Processor (`utils/pdf_processor.py`)
- Extracts text from PDF files
- Splits text into chunks for better processing
- Cleans and preprocesses text

### 2. Vector Store (`utils/vector_store.py`)
- Creates embeddings using Sentence Transformers
- Stores vectors using FAISS for fast similarity search
- Handles document retrieval and similarity matching

### 3. Chat Manager (`utils/chat_manager.py`)
- Manages conversation flow
- Implements RAG (Retrieval-Augmented Generation)
- Integrates with Groq API for response generation
- Logs conversations for analytics

### 4. Settings (`config/settings.py`)
- Handles configuration management
- Provides default settings
- Validates configuration parameters

## 🔒 Security

- **Admin Authentication**: Secure admin panel with password protection
- **API Key Security**: API keys are stored securely and not exposed in UI
- **File Upload Validation**: Only PDF files are allowed
- **Session Management**: Secure session handling

## 📊 Analytics

The application provides analytics including:
- Total conversations
- Chat frequency over time
- Average response time
- Usage patterns
- Document retrieval statistics

## 🛠️ Customization

### Changing Models
1. **Embedding Model**: Update `embedding_model` in configuration
2. **LLM Model**: Update `groq_model` in configuration (must be supported by Groq)

### UI Customization
1. **Colors**: Modify CSS in `app.py`
2. **Layout**: Adjust Streamlit components
3. **Branding**: Update titles and descriptions

### Adding Features
1. **User Authentication**: Implement user login system
2. **Multiple File Types**: Add support for other document types
3. **Advanced Analytics**: Add more detailed metrics
4. **API Endpoints**: Create REST API endpoints

## 🐛 Troubleshooting

### Common Issues

1. **Groq API Key Error**
   - Ensure your API key is valid and has sufficient credits
   - Check that the key starts with 'gsk_'

2. **PDF Processing Fails**
   - Ensure PDF files are not corrupted
   - Check file size limits (default: 10MB)

3. **Vector Store Not Working**
   - Check if `data/vector_store/` directory exists
   - Ensure sufficient disk space

4. **Import Errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Check Python version (3.8+ required)

### Getting Help

1. Check the error messages in the Streamlit interface
2. Look at the terminal output for detailed error logs
3. Verify all dependencies are installed correctly

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For support or questions, please open an issue in the repository or contact the maintainer.

---

**Enjoy using your RAG Chatbot! 🚀**
