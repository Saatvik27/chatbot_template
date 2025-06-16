import json
import os
from datetime import datetime
from typing import List, Dict, Any
from groq import Groq
from utils.vector_store import VectorStore

class ChatManager:
    def __init__(self, chat_history_path: str = "data/chat_history.json"):
        """
        Initialize chat manager
        
        Args:
            chat_history_path: Path to store chat history
        """
        self.chat_history_path = chat_history_path
        self.vector_store = VectorStore()
        
        # Create data directory if it doesn't exist        os.makedirs(os.path.dirname(chat_history_path), exist_ok=True)
        
        # Load existing chat history
        self.chat_history = self._load_chat_history()
    
    def get_response(self, query: str, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile", max_context_length: int = 4000) -> str:
        """
        Get response from Groq model with RAG context
        
        Args:
            query: User's question
            groq_api_key: Groq API key
            model_name: Name of the Groq model to use
            max_context_length: Maximum length of context to include
            
        Returns:
            Generated response
        """
        start_time = datetime.now()
        
        try:
            # Determine if the query needs document context
            needs_context = self._needs_document_context(query)
              # Get relevant context from vector store only if needed
            context_docs = []
            context = ""
            
            if needs_context and self.vector_store.is_initialized():
                context_docs = self.vector_store.search(query, k=5)
                # Only use context if it's actually relevant (good similarity scores)
                relevant_docs = [(text, score) for text, score in context_docs if score > 0.3]
                if relevant_docs:
                    context_docs = relevant_docs
                    context = self._build_context(context_docs, max_context_length)
                else:
                    context_docs = []
                    context = ""
            
            # Debug: Print what we have
            print(f"DEBUG - Query: {query}")
            print(f"DEBUG - Needs context: {needs_context}")
            print(f"DEBUG - Vector store initialized: {self.vector_store.is_initialized()}")
            print(f"DEBUG - Context docs: {len(context_docs)}")
            print(f"DEBUG - Context: {context[:100] if context else 'EMPTY'}")
            
            # Create prompt
            prompt = self._create_prompt(query, context, has_documents=bool(context_docs))
            
            # Get response from Groq
            client = Groq(api_key=groq_api_key)
            
            # Create system message based on whether we have context
            if context_docs:
                system_message = """You are a helpful AI assistant. When provided with context from documents, use that information to answer questions accurately. When no document context is provided, respond naturally as a conversational AI. Always be concise, accurate, and friendly."""
            else:
                system_message = """You are a helpful AI assistant. Respond naturally and conversationally to user messages. Be friendly, concise, and helpful."""
            
            # Handle possible model errors gracefully
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    top_p=1,
                    stream=False
                )
            except Exception as model_error:
                # If the model is decommissioned or unavailable, try a fallback model
                if "model" in str(model_error).lower() and ("decommissioned" in str(model_error).lower() or "unavailable" in str(model_error).lower()):
                    # Try with a fallback model
                    fallback_model = "llama-3.3-70b-versatile"
                    response = client.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.1,
                        max_tokens=1024,
                        top_p=1,
                        stream=False
                    )
                else:
                    # If it's a different error, re-raise it
                    raise
            
            answer = response.choices[0].message.content
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Log the conversation
            self._log_conversation(query, answer, context_docs, response_time)
            
            return answer
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            self._log_conversation(query, error_msg, [], 0)
            return error_msg
    
    def _build_context(self, context_docs: List[tuple], max_length: int) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            context_docs: List of (text, score) tuples
            max_length: Maximum length of context
            
        Returns:
            Context string
        """
        if not context_docs:
            return "No relevant context found in the documents."
        
        context_parts = []
        current_length = 0
        
        for text, score in context_docs:
            # Add document with relevance score
            doc_text = f"[Relevance: {score:.3f}] {text}"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, has_documents: bool = True) -> str:
        """
        Create prompt for the language model
        
        Args:
            query: User's question
            context: Retrieved context or empty string
            has_documents: Whether documents are available
            
        Returns:
            Formatted prompt
        """
        if has_documents and context.strip():
            prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based primarily on the provided context. If the context doesn't contain enough information to fully answer the question, you can supplement with general knowledge, but make it clear what comes from the documents vs. general knowledge."""
        else:
            # For simple queries or when no relevant context is found
            prompt = query
        
        return prompt
    
    def _log_conversation(self, query: str, response: str, context_docs: List[tuple], response_time: float) -> None:
        """
        Log conversation to chat history
        
        Args:
            query: User's question
            response: AI's response
            context_docs: Retrieved context documents
            response_time: Time taken to generate response
        """
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'question': query,
            'response': response,
            'context_count': len(context_docs),
            'context_scores': [score for _, score in context_docs] if context_docs else [],
            'response_time': response_time,
            'user_id': 'anonymous',  # Could be enhanced with user authentication
            'has_documents': bool(context_docs)
        }
        
        self.chat_history.append(conversation)
        self._save_chat_history()
    
    def _load_chat_history(self) -> List[Dict[str, Any]]:
        """Load chat history from file"""
        if os.path.exists(self.chat_history_path):
            try:
                with open(self.chat_history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading chat history: {e}")
                return []
        return []
    
    def _save_chat_history(self) -> None:
        """Save chat history to file"""
        try:
            with open(self.chat_history_path, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def get_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent chat history
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of recent conversations
        """
        return self.chat_history[-limit:] if self.chat_history else []
    
    def clear_chat_history(self) -> None:
        """Clear all chat history"""
        self.chat_history = []
        if os.path.exists(self.chat_history_path):
            os.remove(self.chat_history_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat statistics
        
        Returns:
            Dictionary with chat statistics
        """
        if not self.chat_history:
            return {
                'total_conversations': 0,
                'average_response_time': 0,
                'average_context_score': 0,
                'document_based_chats': 0,
                'general_chats': 0
            }
        
        total_conversations = len(self.chat_history)
        
        # Calculate average response time
        response_times = [conv.get('response_time', 0) for conv in self.chat_history]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate average context score
        all_scores = []
        for conv in self.chat_history:
            scores = conv.get('context_scores', [])
            all_scores.extend(scores)
        
        avg_context_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Count document-based vs general chats
        document_based = sum(1 for conv in self.chat_history if conv.get('has_documents', False))
        general_chats = total_conversations - document_based
        
        return {
            'total_conversations': total_conversations,
            'average_response_time': avg_response_time,
            'average_context_score': avg_context_score,
            'document_based_chats': document_based,
            'general_chats': general_chats,
            'unique_days': len(set(conv['timestamp'][:10] for conv in self.chat_history))
        }
    
    def _needs_document_context(self, query: str) -> bool:
        """
        Determine if a query needs document context based on its content
        
        Args:
            query: User's question
            
        Returns:
            True if the query likely needs document context, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Simple greetings and conversational phrases that don't need context
        simple_phrases = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what\'s up', 'thanks', 'thank you', 'bye', 'goodbye',
            'ok', 'okay', 'yes', 'no', 'sure', 'alright', 'cool', 'nice'
        ]
        
        # Check if the query is just a simple greeting/phrase
        if query_lower in simple_phrases:
            return False
        
        # Check if it's a very short query (likely greeting)
        if len(query_lower.split()) <= 2 and any(phrase in query_lower for phrase in simple_phrases):
            return False
        
        # Questions that typically need document context
        context_indicators = [
            'what is', 'what are', 'how to', 'explain', 'describe', 'tell me about',
            'definition of', 'meaning of', 'according to', 'based on', 'document',
            'text', 'chapter', 'section', 'page', 'book', 'paper', 'study'
        ]
        
        # If it contains context indicators, it likely needs documents
        if any(indicator in query_lower for indicator in context_indicators):
            return True
        
        # For medium to long queries, assume they might need context
        if len(query_lower.split()) >= 5:
            return True
        
        # Default: don't use context for short, simple queries
        return False
