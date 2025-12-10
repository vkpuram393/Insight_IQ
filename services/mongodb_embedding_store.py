"""
MongoDB Embedding Store Service

Handles storage and retrieval of intent embeddings in MongoDB.
Uses nested structure: one document per intent with embedded examples array.

Collections:
- intent_embeddings: 30 documents (one per intent)
- embedding_metadata: 1 document (cache info and hashes)
"""

import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.logger import get_logger

logger = get_logger(__name__)


class MongoDBEmbeddingStoreFactory:
    """Factory to get singleton instance of MongoDBEmbeddingStore (like team's PersistenceStoreFactory)"""
    
    _instance: "MongoDBEmbeddingStore" = None
    
    @classmethod
    def get_instance(cls) -> "MongoDBEmbeddingStore":
        """Get singleton instance of embedding store"""
        if cls._instance is None:
            cls._instance = MongoDBEmbeddingStore()
            logger.info("🔄 Created singleton MongoDBEmbeddingStore instance")
        return cls._instance
    
    @classmethod
    async def close_instance(cls) -> None:
        """Close the current instance"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None


class MongoDBEmbeddingStore:
    """MongoDB-based storage for intent embeddings"""
    
    def __init__(self, connection_string: str = None, database_name: str = None):
        """
        Initialize MongoDB connection for embeddings
        
        Args:
            connection_string: MongoDB connection string (default from config)
            database_name: Database name (default from config)
        """
        from config.config import settings
        
        self.connection_string = connection_string or settings.mongodb_connection_string
        self.database_name = database_name or settings.mongodb_database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        
        logger.info(f"📊 MongoDBEmbeddingStore initialized: {self.database_name}")
    
    async def _get_connection(self):
        """Get or create MongoDB connection"""
        if self.client is None:
            try:
                # SSL certificate fix for macOS (uncomment if needed)
                # import certifi
                # tls_ca_file = certifi.where()
                
                self.client = AsyncIOMotorClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    retryWrites=True,
                    # tlsCAFile=tls_ca_file  # Uncomment if using certifi above
                )
                # Test connection
                await self.client.admin.command('ping')
                self.db = self.client[self.database_name]
                logger.info(f"✅ MongoDB connection established: {self.database_name}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
                raise
        return self.db
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("🔌 MongoDB connection closed")
    
    @staticmethod
    def calculate_examples_hash(intent_examples: Dict[str, list]) -> str:
        """
        Calculate hash of all intent examples to detect changes
        
        Args:
            intent_examples: Dict of intent -> list of example texts
            
        Returns:
            SHA256 hash of examples (deterministic)
        """
        # Create deterministic JSON string
        examples_str = json.dumps(intent_examples, sort_keys=True)
        hash_obj = hashlib.sha256(examples_str.encode())
        return hash_obj.hexdigest()
    
    async def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get embedding metadata from MongoDB
        
        Returns:
            Metadata dict or None if not found
        """
        try:
            db = await self._get_connection()
            metadata = await db.embedding_metadata.find_one({"_id": "cache_info"})
            return metadata
        except Exception as e:
            logger.error(f"❌ Failed to get metadata: {str(e)}")
            return None
    
    async def check_cache_validity(
        self,
        current_provider: str,
        current_dimension: int,
        current_examples: Dict[str, list]
    ) -> bool:
        """
        Check if cached embeddings in MongoDB are still valid
        
        Args:
            current_provider: Current embedding provider (e.g. "Google Cloud")
            current_dimension: Current embedding dimension (e.g. 768)
            current_examples: Current intent examples from config
            
        Returns:
            True if cache is valid, False if regeneration needed
        """
        metadata = await self.get_metadata()
        
        if not metadata:
            logger.info("🔍 No metadata found - cache invalid")
            return False
        
        # Check provider
        if metadata.get("embedding_provider") != current_provider:
            logger.warning(
                f"⚠️  Provider mismatch: "
                f"cached={metadata.get('embedding_provider')}, current={current_provider}"
            )
            return False
        
        # Check dimension
        if metadata.get("embedding_dimension") != current_dimension:
            logger.warning(
                f"⚠️  Dimension mismatch: "
                f"cached={metadata.get('embedding_dimension')}, current={current_dimension}"
            )
            return False
        
        # Check examples hash
        current_hash = self.calculate_examples_hash(current_examples)
        cached_hash = metadata.get("examples_hash")
        
        if cached_hash != current_hash:
            logger.warning(
                f"⚠️  Examples hash mismatch:\n"
                f"   Cached: {cached_hash[:16]}...\n"
                f"   Current: {current_hash[:16]}..."
            )
            logger.info("   → Examples have changed, regeneration needed")
            return False
        
        logger.info("✅ Cache is valid (provider, dimension, and examples match)")
        return True
    
    async def load_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all intent embeddings from MongoDB (FLAT structure)
        
        Loads from flat structure where each document represents one example.
        Groups by intent and converts to numpy arrays.
        
        Returns:
            Dict mapping intent -> numpy array of embeddings (shape: [num_examples, embedding_dim])
            or None if no embeddings found
        """
        try:
            db = await self._get_connection()
            
            # Load all embedding documents (flat structure)
            cursor = db.intent_embeddings.find({})
            docs = await cursor.to_list(length=None)
            
            if not docs:
                logger.info("🔍 No embeddings found in MongoDB")
                return None
            
            logger.info(f"📥 Loading embeddings from MongoDB ({len(docs)} documents)...")
            
            # Group by intent (since it's flat structure now)
            intent_groups = {}
            for doc in docs:
                intent = doc.get("intent")
                embedding = doc.get("embedding")
                
                if not intent or not embedding:
                    logger.warning(f"⚠️  Skipping invalid document: {doc.get('_id')}")
                    continue
                
                if intent not in intent_groups:
                    intent_groups[intent] = []
                
                intent_groups[intent].append(embedding)
            
            # Convert to numpy arrays
            intent_embeddings = {}
            for intent, embeddings_list in intent_groups.items():
                intent_embeddings[intent] = np.array(embeddings_list, dtype=np.float32)
                
                logger.debug(
                    f"   Loaded {intent}: {len(embeddings_list)} examples × "
                    f"{len(embeddings_list[0])} dimensions"
                )
            
            logger.info(f"✅ Loaded {len(intent_embeddings)} intents from MongoDB (INSTANT - no API calls!)")
            return intent_embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings from MongoDB: {str(e)}")
            return None
    
    async def save_embeddings(
        self,
        intent_embeddings: Dict[str, np.ndarray],
        intent_examples: Dict[str, list],
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int
    ):
        """
        Save all intent embeddings to MongoDB (FLAT structure for vector search)
        
        Creates ONE document per example (not nested) so MongoDB Atlas Vector Search
        can index the embedding field directly for fast similarity search.
        
        Args:
            intent_embeddings: Dict mapping intent -> numpy array of embeddings
            intent_examples: Dict mapping intent -> list of example texts
            embedding_provider: Provider name (e.g. "Google Cloud")
            embedding_model: Model name (e.g. "text-embedding-005")
            embedding_dimension: Embedding dimension (e.g. 768)
        """
        try:
            db = await self._get_connection()
            
            logger.info("🔄 Saving embeddings to MongoDB (nested structure)...")
            
            # Step 1: Delete all existing embeddings (clean slate)
            result = await db.intent_embeddings.delete_many({})
            if result.deleted_count > 0:
                logger.info(f"🗑️  Deleted {result.deleted_count} old embedding documents")
            
            # Step 2: Prepare documents (FLAT: one per example for vector search)
            documents = []
            total_examples = 0
            
            logger.info("📝 Creating FLAT structure (1 document per example for vector search)...")
            
            for intent, embeddings_array in intent_embeddings.items():
                examples_list = intent_examples.get(intent, [])
                
                if len(examples_list) != len(embeddings_array):
                    logger.warning(
                        f"⚠️  Mismatch for '{intent}': "
                        f"{len(examples_list)} texts but {len(embeddings_array)} embeddings"
                    )
                    continue
                
                # Create ONE document per example (flat structure)
                for idx, (text, embedding) in enumerate(zip(examples_list, embeddings_array)):
                    doc = {
                        "intent": intent,
                        "text": text,
                        "embedding": embedding.tolist(),  # Vector index on this field
                        "index": idx,
                        "embedding_dimension": embedding_dimension,
                        "provider": embedding_provider,
                        "model": embedding_model,
                        "created_at": datetime.utcnow()
                    }
                    documents.append(doc)
                    total_examples += 1
                
                logger.debug(f"   Prepared {intent}: {len(embeddings_array)} examples")
            
            # Step 3: Insert all documents (batch insert)
            if documents:
                await db.intent_embeddings.insert_many(documents)
                logger.info(
                    f"✅ Inserted {len(documents)} intent documents "
                    f"({total_examples} total examples) to MongoDB"
                )
            
            # Step 4: Update metadata
            examples_hash = self.calculate_examples_hash(intent_examples)
            
            metadata = {
                "_id": "cache_info",
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "total_intents": len(intent_embeddings),
                "total_examples": total_examples,
                "examples_hash": examples_hash,
                "structure": "flat",  # Flat structure for vector search
                "vector_search_ready": True,  # Ready for Atlas Vector Search
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await db.embedding_metadata.replace_one(
                {"_id": "cache_info"},
                metadata,
                upsert=True
            )
            
            logger.info(
                f"✅ Updated metadata: {embedding_provider}, "
                f"{embedding_dimension}-dim, hash={examples_hash[:16]}..."
            )
            logger.info("🎉 Embeddings successfully saved to MongoDB!")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings to MongoDB: {str(e)}")
            raise
    
    async def vector_search(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 50
    ) -> Dict[str, float]:
        """
        Perform MongoDB Atlas Vector Search to find most similar examples
        
        This uses MongoDB's native vector search with cosine similarity
        to find the most similar examples across all intents.
        
        Args:
            query_embedding: Query embedding vector (numpy array)
            limit: Maximum number of results to return (default 50)
            
        Returns:
            Dict mapping intent -> max similarity score
        """
        try:
            db = await self._get_connection()
            
            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Name of vector search index
                        "path": "embedding",      # Field containing the vector
                        "queryVector": query_embedding.tolist(),  # Convert numpy to list
                        "numCandidates": limit * 5,  # Candidates to consider
                        "limit": limit  # Max results to return
                    }
                },
                {
                    "$project": {
                        "intent": 1,
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"}  # Get similarity score
                    }
                }
            ]
            
            # Execute vector search
            cursor = db.intent_embeddings.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            if not results:
                logger.warning("⚠️  Vector search returned no results")
                return {}
            
            # Group by intent and take max score for each intent
            intent_scores = {}
            for doc in results:
                intent = doc.get("intent")
                score = doc.get("score", 0.0)
                
                if intent:
                    # Keep the highest score for each intent
                    if intent not in intent_scores or score > intent_scores[intent]:
                        intent_scores[intent] = float(score)
            
            logger.debug(f"🔍 Vector search found {len(results)} results across {len(intent_scores)} intents")
            
            return intent_scores
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for common errors
            if "index not found" in error_msg.lower() or "vector" in error_msg.lower():
                logger.error(
                    f"❌ Vector search index not found or not ready. "
                    f"Run: python scripts/setup_vector_index.py"
                )
            else:
                logger.error(f"❌ Vector search failed: {error_msg}")
            
            raise
    
    async def clear_embeddings(self):
        """Delete all embeddings and metadata from MongoDB"""
        try:
            db = await self._get_connection()
            
            # Delete all embeddings
            result1 = await db.intent_embeddings.delete_many({})
            logger.info(f"🗑️  Deleted {result1.deleted_count} embedding documents")
            
            # Delete metadata
            result2 = await db.embedding_metadata.delete_many({})
            logger.info(f"🗑️  Deleted {result2.deleted_count} metadata documents")
            
            logger.info("✅ MongoDB embeddings cleared")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear embeddings: {str(e)}")
            raise
