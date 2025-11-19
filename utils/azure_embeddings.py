"""
Azure OpenAI Embeddings Service
Supports both API Key and Azure AD (Service Principal) authentication
Adapted from myclaim-chatbot/backend/services/azure_embeddings.py
"""

from typing import List, Union
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

# Global singleton
_azure_embeddings = None


class AzureEmbeddings:
    """Azure OpenAI Embeddings with Azure AD support (CVS-compatible pattern)"""
    
    def __init__(self):
        """Initialize Azure OpenAI client with appropriate authentication"""
        from core.config import settings
        
        # Check if using Azure AD or API Key
        if hasattr(settings, 'azure_tenant_id') and settings.azure_tenant_id:
            logger.info("Using Azure AD with DefaultAzureCredential for embeddings (CVS pattern)")
            self._init_with_azure_ad_cvs_pattern(settings)
        elif hasattr(settings, 'azure_openai_key') and settings.azure_openai_key:
            logger.info("Using Azure OpenAI API Key authentication for embeddings")
            self._init_with_api_key(settings)
        else:
            logger.warning(
                "⚠️  Azure OpenAI Embeddings not configured. Using mock embeddings.\n"
                "   To enable, set in core/config.py or .env:\n"
                "     1. Azure AD: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET\n"
                "     2. API Key: AZURE_OPENAI_KEY"
            )
            self.client = None
            self.auth_method = "Mock (not configured)"
            return
        
        logger.info(f"✅ Azure OpenAI Embeddings initialized successfully")
        if hasattr(settings, 'azure_openai_endpoint'):
            logger.info(f"   Endpoint: {settings.azure_openai_endpoint}")
        if hasattr(settings, 'azure_openai_embedding_model'):
            logger.info(f"   Deployment: {settings.azure_openai_embedding_model}")
        if hasattr(settings, 'azure_openai_api_version'):
            logger.info(f"   API Version: {settings.azure_openai_api_version}")
    
    def _init_with_azure_ad_cvs_pattern(self, settings):
        """
        Initialize with Azure AD using DefaultAzureCredential (CVS Pattern)
        Matches CVS's nlx_qa_agent.py implementation
        """
        try:
            from azure.identity import DefaultAzureCredential
            from openai import AzureOpenAI
            
            # Export credentials to environment for DefaultAzureCredential
            os.environ['AZURE_TENANT_ID'] = settings.azure_tenant_id
            os.environ['AZURE_CLIENT_ID'] = settings.azure_client_id
            os.environ['AZURE_CLIENT_SECRET'] = settings.azure_client_secret
            
            logger.info("Exported Azure credentials to environment")
            logger.info(f"Endpoint: {settings.azure_openai_endpoint}")
            logger.info(f"Deployment: {settings.azure_openai_embedding_model}")
            
            # Get credentials from environment (CVS pattern)
            credential = DefaultAzureCredential()
            logger.info("DefaultAzureCredential created")
            
            # Get token manually (CVS pattern)
            access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
            logger.info("✅ Access token obtained successfully!")
            
            # Initialize Azure OpenAI client with static token (CVS pattern)
            self.client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_ad_token=access_token.token
            )
            self.auth_method = "Azure AD (CVS Pattern - DefaultAzureCredential)"
            logger.info("✅ Azure OpenAI Embeddings client initialized")
        except ImportError as e:
            logger.warning(f"⚠️  Azure dependencies not installed: {e}")
            logger.warning("   Install with: pip install azure-identity openai")
            self.client = None
            self.auth_method = "Mock (missing dependencies)"
        except Exception as e:
            logger.error(f"❌ Azure AD authentication failed: {e}")
            self.client = None
            self.auth_method = "Mock (auth failed)"
    
    def _init_with_api_key(self, settings):
        """Initialize with API Key"""
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version
            )
            self.auth_method = "API Key"
        except ImportError as e:
            logger.warning(f"⚠️  openai library not installed: {e}")
            logger.warning("   Install with: pip install openai")
            self.client = None
            self.auth_method = "Mock (missing dependencies)"
        except Exception as e:
            logger.error(f"❌ API Key authentication failed: {e}")
            self.client = None
            self.auth_method = "Mock (auth failed)"
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding(s) for text using Azure OpenAI
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Single vector or list of vectors
        """
        # If not configured, return mock embeddings
        if self.client is None:
            logger.debug("Using mock embeddings (Azure not configured)")
            is_single = isinstance(text, str)
            dimensions = 1536  # Default for text-embedding-ada-002
            
            if is_single:
                return list(np.random.rand(dimensions).astype(float))
            else:
                return [list(np.random.rand(dimensions).astype(float)) for _ in text]
        
        try:
            from core.config import settings
            
            # Convert single string to list for uniform processing
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Call Azure OpenAI Embeddings API
            response = self.client.embeddings.create(
                model=settings.azure_openai_embedding_model,  # deployment name
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Return single vector or list based on input type
            return embeddings[0] if is_single else embeddings
        
        except Exception as e:
            logger.error(f"Azure OpenAI embedding generation failed: {e}")
            logger.warning("Falling back to mock embeddings")
            # Return random vector as fallback (for testing)
            dimensions = 1536
            is_single = isinstance(text, str)
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
        """
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


def get_azure_embeddings() -> AzureEmbeddings:
    """Get global Azure Embeddings instance (singleton pattern)"""
    global _azure_embeddings
    if _azure_embeddings is None:
        _azure_embeddings = AzureEmbeddings()
    return _azure_embeddings


def get_embedding(text: str) -> List[float]:
    """
    Convenience function to get a single embedding
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector (1536 dimensions)
    """
    embeddings = get_azure_embeddings()
    return embeddings.embed(text)

