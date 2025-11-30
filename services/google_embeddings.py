"""
Google Cloud Vertex AI Embeddings Service
Uses text-embedding-005 model (same auth pattern as Gemini LLM)
"""

from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global singleton
_google_embeddings = None


class GoogleEmbeddings:
    """Google Cloud Vertex AI Embeddings using text-embedding-005"""
    
    def __init__(self):
        """Initialize Google Cloud Vertex AI client (same pattern as LLM)"""
        from config.config import settings
        
        # Use same project_id and location as Gemini LLM
        project_id = settings.project_id
        location = settings.location
        
        if not project_id:
            logger.warning(
                "⚠️  Google Cloud Vertex AI Embeddings not configured. Using mock embeddings.\n"
                "   Project ID is not set in config.py"
            )
            self.client = None
            self.auth_method = "Mock (not configured)"
            return
        
        logger.info(f"Using Google Cloud Vertex AI for embeddings (same auth as Gemini LLM)")
        self._init_with_gcp(project_id, location)
        
        if self.client:
            logger.info(f"✅ Google Cloud Vertex AI Embeddings initialized successfully")
            logger.info(f"   Project: {project_id}")
            logger.info(f"   Region: {location}")
            logger.info(f"   Model: text-embedding-005")
    
    def _init_with_gcp(self, project_id, location):
        """Initialize with Google Cloud Vertex AI (same pattern as Gemini LLM)"""
        try:
            from google import genai
            from google.genai import types
            
            logger.info(f"🔐 Initializing Google GenAI client in {project_id} ({location})")
            logger.info("   Using same authentication as Gemini LLM")
            
            # Create client with same pattern as llm_connection.py
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )
            self.model_name = "text-embedding-005"  # Google's latest text embedding model
            self.auth_method = "Google Cloud Application Default Credentials (same as LLM)"
            self.project_id = project_id
            self.region = location
            
            logger.info("✅ Google GenAI client initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️  Google GenAI SDK not installed: {e}")
            logger.warning("   Install with: pip install google-genai")
            self.client = None
            self.auth_method = "Mock (missing dependencies)"
        except Exception as e:
            logger.error(f"❌ Google Cloud authentication failed: {e}")
            logger.error(f"   Error details: {str(e)}")
            logger.error("   Make sure you've run: gcloud auth application-default login")
            self.client = None
            self.auth_method = "Mock (auth failed)"
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding(s) for text using Google Cloud Vertex AI
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Single vector or list of vectors (768 dimensions for text-embedding-005)
        """
        # If not configured, return mock embeddings
        if self.client is None:
            logger.debug("Using mock embeddings (Google Cloud not configured)")
            is_single = isinstance(text, str)
            dimensions = 768  # text-embedding-005 dimensions
            
            if is_single:
                return list(np.random.rand(dimensions).astype(float))
            else:
                return [list(np.random.rand(dimensions).astype(float)) for _ in text]
        
        try:
            from google.genai import types
            
            # Convert single string to list for uniform processing
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Generate embeddings using Google GenAI SDK (same as LLM)
            embeddings_list = []
            for text_item in texts:
                # Create embedding request
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=[types.Part.from_text(text=text_item)],
                )
                # Extract embedding vector
                embeddings_list.append(response.embeddings[0].values)
            
            # Return single vector or list based on input type
            return embeddings_list[0] if is_single else embeddings_list
        
        except Exception as e:
            logger.error(f"Google Cloud Vertex AI embedding generation failed: {e}")
            logger.warning("Falling back to mock embeddings")
            # Return random vector as fallback (for testing)
            dimensions = 768
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            if is_single:
                return list(np.random.rand(dimensions).astype(float))
            else:
                return [list(np.random.rand(dimensions).astype(float)) for _ in texts]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(
        self,
        query_vector: List[float],
        candidate_vectors: List[List[float]]
    ) -> List[float]:
        """
        Calculate cosine similarity between query and multiple candidates
        
        Args:
            query_vector: Query embedding vector
            candidate_vectors: List of candidate embedding vectors
            
        Returns:
            List of similarity scores
            
        Raises:
            ValueError: If dimension mismatch detected (indicates cache needs regeneration)
        """
        # Validate dimensions before processing
        query_dim = len(query_vector) if query_vector else 0
        if candidate_vectors and len(candidate_vectors) > 0:
            candidate_dim = len(candidate_vectors[0]) if candidate_vectors[0] else 0
            if query_dim != candidate_dim and query_dim > 0 and candidate_dim > 0:
                raise ValueError(
                    f"Embedding dimension mismatch: query={query_dim}, cached={candidate_dim}. "
                    f"This usually means the cache was created with a different embedding provider. "
                    f"Delete 'classifiers/intent_embeddings_cache.pkl' and restart the server to regenerate."
                )
        
        return [
            self.cosine_similarity(query_vector, candidate)
            for candidate in candidate_vectors
        ]
    
    def search_similar(
        self,
        query_vector: List[float],
        candidate_vectors: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Search for most similar vectors
        
        Args:
            query_vector: Query embedding
            candidate_vectors: Candidates to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, score) tuples
        """
        similarities = self.batch_similarity(query_vector, candidate_vectors)
        
        # Filter by threshold and get top-k
        results = [
            (idx, score)
            for idx, score in enumerate(similarities)
            if score >= threshold
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


def get_google_embeddings() -> GoogleEmbeddings:
    """Get global Google Embeddings instance (singleton pattern)"""
    global _google_embeddings
    if _google_embeddings is None:
        _google_embeddings = GoogleEmbeddings()
    return _google_embeddings


def get_embedding(text: str) -> List[float]:
    """
    Convenience function to get a single embedding
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector (1408 dimensions)
    """
    embeddings = get_google_embeddings()
    return embeddings.embed(text)

