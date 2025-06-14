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
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(chat_history_path), exist_ok=True)
        
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
            # Get relevant context from vector store (if available)
            context_docs = []
            context = ""
            
            if self.vector_store.is_initialized():
                context_docs = self.vector_store.search(query, k=5)
                context = self._build_context(context_docs, max_context_length)
            else:
                context = "No documents are available in the knowledge base."
            
            # Create prompt
            prompt = self._create_prompt(query, context, has_documents=bool(context_docs))
              # Get response from Groq
            client = Groq(api_key=groq_api_key)
            
            # Handle possible model errors gracefully
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a helpful AI assistant. If context from documents is provided, use it to answer questions. If no documents are available, provide helpful general answers. Always be concise, accurate, and friendly in your responses."""
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
                                "content": """You are a helpful AI assistant. If context from documents is provided, use it to answer questions. If no documents are available, provide helpful general answers. Always be concise, accurate, and friendly in your responses."""
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
        if has_documents:
            prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based primarily on the provided context. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing and provide any general knowledge that might be helpful."""
        else:
            prompt = f"""Question: {query}

Note: No specific documents are available for reference, so please provide a helpful answer based on your general knowledge. Be informative, accurate, and helpful in your response."""
        
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
