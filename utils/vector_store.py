import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
import json
from pathlib import Path

# Lazy import to avoid torch issues at startup
def get_sentence_transformer():
    try:
        # Import torch with error handling
        import torch
        torch.set_num_threads(1)  # Reduce threading issues
        
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as e:
        raise ImportError(f"Please install sentence-transformers: pip install sentence-transformers") from e
    except Exception as e:
        print(f"Warning: torch setup issue: {e}")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_store_path: str = "data/vector_store"):
        """
        Initialize vector store with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
            vector_store_path: Path to store vector database files
        """
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.index_path = os.path.join(vector_store_path, "faiss_index.bin")
        self.texts_path = os.path.join(vector_store_path, "texts.pkl")
        self.metadata_path = os.path.join(vector_store_path, "metadata.json")
          # Create directory if it doesn't exist
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize model lazily
        self.model = None
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = None
        self.texts = []
        self.metadata = {}
        
        # Load existing vector store if available
        self._load_vector_store()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model lazily"""
        if self.model is None:
            try:
                SentenceTransformer = get_sentence_transformer()
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"✅ Loaded embedding model: {self.model_name}")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
                raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def create_vector_store(self, texts: List[str]) -> None:
        """
        Create vector store from list of texts
        
        Args:
            texts: List of text chunks to embed
        """
        if not texts:
            raise ValueError("No texts provided to create vector store")
        
        # Initialize model if not done yet
        self._initialize_model()
        
        print(f"Creating embeddings for {len(texts)} text chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store texts
        self.texts = texts
          # Update metadata
        from datetime import datetime
        self.metadata = {
            'total_documents': len(texts),
            'model_name': self.model_name,
            'dimension': self.dimension,
            'created_at': datetime.now().isoformat()
        }
        
        # Save vector store
        self._save_vector_store()
        
        print(f"Vector store created successfully with {len(texts)} documents!")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of top results to return
            
        Returns:
            List of tuples (text, similarity_score)
        """
        if not self.is_initialized():
            return []
        
        # Initialize model if not done yet
        self._initialize_model()
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):  # Valid index
                results.append((self.texts[idx], float(score)))
        
        return results
    
    def add_documents(self, texts: List[str]) -> None:
        """
        Add new documents to existing vector store
        
        Args:
            texts: List of new text chunks to add
        """
        if not texts:
            return
        
        # Initialize model if not done yet
        self._initialize_model()
        
        # Generate embeddings for new texts
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        if self.index is None:
            # Create new index if none exists
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add to texts
        self.texts.extend(texts)
          # Update metadata
        from datetime import datetime
        self.metadata['total_documents'] = len(self.texts)
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        # Save updated vector store
        self._save_vector_store()
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized and has data"""
        return self.index is not None and len(self.texts) > 0
    
    def get_info(self) -> dict:
        """Get information about the vector store"""
        return {
            'total_documents': len(self.texts),
            'model_name': self.model_name,
            'dimension': self.dimension,
            'is_initialized': self.is_initialized(),
            'metadata': self.metadata
        }
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.index = None
        self.texts = []
        self.metadata = {}
        
        # Remove saved files
        for path in [self.index_path, self.texts_path, self.metadata_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def _save_vector_store(self) -> None:
        """Save vector store to disk"""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            
            # Save texts
            with open(self.texts_path, 'wb') as f:
                pickle.dump(self.texts, f)
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
    
    def _load_vector_store(self) -> None:
        """Load vector store from disk"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            
            # Load texts
            if os.path.exists(self.texts_path):
                with open(self.texts_path, 'rb') as f:
                    self.texts = pickle.load(f)
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            # Reset on error
            self.index = None
            self.texts = []
            self.metadata = {}
