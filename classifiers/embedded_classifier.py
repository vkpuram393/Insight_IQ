"""
CVS Intent Classifier - Embedding-Based (Zero-Shot)
Uses Azure OpenAI embeddings + cosine similarity for intent classification
NO TRAINING required - semantic understanding through embeddings

This classifier returns the same structure as keyword_classifier.py
so it can be used as a drop-in replacement for A/B testing.

Comparison:
- keyword_classifier.py: Keyword-based (fast, rule-based)
- embedded_classifier.py: Embedding-based (semantic understanding)
"""

import numpy as np
import re
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

# Import embedding service (dynamic based on config)
# This happens at module load time, so changes require server restart
try:
    from config.config import settings
    
    if getattr(settings, 'use_google_embeddings', False):
        # Use Google Cloud Vertex AI embeddings
        from services.google_embeddings import get_embedding, get_google_embeddings as get_embeddings_service
        EMBEDDINGS_AVAILABLE = True
        EMBEDDINGS_PROVIDER = "Google Cloud Vertex AI"
        logging.info("🟢 Using Google Cloud Vertex AI embeddings for runtime queries")
    else:
        # Use Azure OpenAI embeddings (default)
        from services.azure_embeddings import get_embedding, get_azure_embeddings as get_embeddings_service
        EMBEDDINGS_AVAILABLE = True
        EMBEDDINGS_PROVIDER = "Azure OpenAI"
        logging.info("🔵 Using Azure OpenAI embeddings for runtime queries")
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    EMBEDDINGS_PROVIDER = "Mock"
    logging.warning(f"Embeddings not available: {e}. Using mock embeddings.")

logger = logging.getLogger(__name__)

# Global singleton instance to prevent memory leaks
# Creating new instances loads all embeddings into memory (~800MB)
_embedded_classifier_instance = None


def get_embedded_classifier() -> "CVSIntentEmbedded":
    """Get global embedded classifier instance (singleton pattern)"""
    global _embedded_classifier_instance
    if _embedded_classifier_instance is None:
        _embedded_classifier_instance = CVSIntentEmbedded()
        logger.info("🔄 Created singleton CVSIntentEmbedded instance")
    return _embedded_classifier_instance


class CVSIntentEmbedded:
    """
    Embedding-based intent classifier using semantic similarity
    
    Returns same structure as CVSIntentClassifier for compatibility:
    {
        'intent': str,
        'confidence': float,
        'all_scores': dict,
        'is_simple': bool,
        'is_complex': bool,
        'needs_clarification': bool
    }
    """
    
    def __init__(self):
        """Initialize with embedded intent examples"""
        logger.info("Initializing CVS Intent Classifier (Embedding-Based)...")
        
        # Cache the embeddings service (singleton) - needed before _embed_all_examples
        if EMBEDDINGS_AVAILABLE:
            self.embeddings_service = get_embeddings_service()
            logger.info(f"✅ Using {EMBEDDINGS_PROVIDER} for intent embeddings")
        else:
            self.embeddings_service = None
            logger.warning("⚠️ Using mock embeddings (no provider configured)")
        
        # Build intent examples (embedded in this file)
        self.intent_examples = self._build_intent_examples()
        
        # Check if MongoDB vector search is enabled
        from config.config import settings
        if settings.use_mongodb_for_embeddings:
            # MongoDB Vector Search mode: Ensure embeddings exist in MongoDB (auto-generate if empty)
            import asyncio
            import concurrent.futures
            
            def ensure_embeddings():
                return asyncio.run(self._embed_all_examples())
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ensure_embeddings)
                future.result()  # No timeout - startup controls timing
            
            logger.info("🚀 MongoDB Vector Search enabled - embeddings ready, will query on-demand")
            self.intent_embeddings = {}  # Empty dict, queries go to MongoDB
        else:
            # Traditional mode: Pre-load all embeddings into memory
            # Use ThreadPoolExecutor to run async code in separate thread with new event loop
            import asyncio
            import concurrent.futures
            
            def run_embed_in_thread():
                """Run async embedding in new thread with its own event loop"""
                return asyncio.run(self._embed_all_examples())
            
            # Always use ThreadPoolExecutor to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_embed_in_thread)
                self.intent_embeddings = future.result()  # No timeout - startup controls timing
        
        # Thresholds
        self.confidence_threshold = 0.50  # Match keyword classifier
        
        logger.info(f"✅ CVS Intent Classifier (Embeddings) initialized")
        logger.info(f"   Intents: {len(self.intent_examples)}")
        logger.info(f"   Total examples: {sum(len(ex) for ex in self.intent_examples.values())}")
    
    def _build_intent_examples(self) -> Dict[str, List[str]]:
        """
        Build intent examples (LLM-generated, no data leakage)
        
        600 examples total (20 per intent, 30 intents)
        Generated from real CVS queries but EXCLUDING test queries
        
        Returns:
            Dict mapping intent name to list of example queries
        """
        return CVS_INTENT_EXAMPLES
    
    def _get_expected_embedding_dimension(self) -> int:
        """
        Get the expected embedding dimension for the current embedding provider
        
        Returns:
            Expected dimension (768 for Google, 1536 for Azure)
        """
        # Try to get dimension from a test embedding (most reliable)
        if EMBEDDINGS_AVAILABLE and self.embeddings_service is not None:
            try:
                test_embedding = get_embedding("test")
                if isinstance(test_embedding, (list, np.ndarray)):
                    dim = len(test_embedding)
                    logger.debug(f"📏 Detected embedding dimension: {dim} (from test embedding)")
                    return dim
            except Exception as e:
                logger.debug(f"Could not get test embedding for dimension check: {e}")
        
        # Fallback: return dimension based on provider
        if EMBEDDINGS_PROVIDER == "Google Cloud Vertex AI":
            return 768  # text-embedding-005
        elif EMBEDDINGS_PROVIDER == "Azure OpenAI":
            return 1536  # text-embedding-ada-002
        else:
            return 1536  # Default to Azure dimension
    
    def _validate_cache_dimensions(self, intent_embeddings: Dict[str, np.ndarray]) -> bool:
        """
        Validate that cached embeddings match current embedding provider dimensions
        
        Args:
            intent_embeddings: Cached embeddings dict
            
        Returns:
            True if dimensions match, False otherwise
        """
        if not intent_embeddings:
            return False
        
        # Get expected dimension
        expected_dim = self._get_expected_embedding_dimension()
        
        # Check first embedding to get actual dimension
        first_intent = next(iter(intent_embeddings.values()))
        if isinstance(first_intent, np.ndarray) and len(first_intent.shape) >= 1:
            actual_dim = first_intent.shape[-1]  # Last dimension is the embedding size
        elif isinstance(first_intent, list) and len(first_intent) > 0:
            first_embedding = first_intent[0]
            if isinstance(first_embedding, (list, np.ndarray)):
                actual_dim = len(first_embedding)
            else:
                return False
        else:
            return False
        
        if actual_dim != expected_dim:
            logger.warning(
                f"⚠️  Cache dimension mismatch: cached={actual_dim}, expected={expected_dim} "
                f"(provider={EMBEDDINGS_PROVIDER})"
            )
            return False
        
        return True
    
    async def _embed_all_examples(self) -> Dict[str, np.ndarray]:
        """
        Convert all intent examples to embeddings
        
        Priority:
        1. Try MongoDB first (if enabled)
        2. Fall back to .pkl file cache
        3. Generate fresh if neither available
        
        Automatically regenerates if dimension/provider mismatch detected.
        
        Returns:
            Dict mapping intent name to array of embeddings
        """
        import pickle
        import os
        from config.config import settings
        
        # ========================================================================
        # PRIORITY 1: Try MongoDB first (if enabled)
        # ========================================================================
        if settings.use_mongodb_for_embeddings:
            try:
                from services.mongodb_embedding_store import MongoDBEmbeddingStore
                
                logger.info("🔍 Checking MongoDB for cached embeddings...")
                mongo_store = MongoDBEmbeddingStore()
                
                # Check if cache is valid
                is_valid = await mongo_store.check_cache_validity(
                    current_provider=EMBEDDINGS_PROVIDER,
                    current_dimension=self._get_expected_embedding_dimension(),
                    current_examples=self.intent_examples
                )
                
                if is_valid:
                    # Load from MongoDB
                    intent_embeddings = await mongo_store.load_embeddings()
                    
                    if intent_embeddings:
                        await mongo_store.close()
                        logger.info("✅ Loaded embeddings from MongoDB (INSTANT - no API calls!)")
                        return intent_embeddings
                
                # Cache invalid or empty - will regenerate and save to MongoDB later
                await mongo_store.close()
                logger.info("⚠️  MongoDB cache invalid or empty, will generate fresh embeddings")
                
            except Exception as e:
                logger.warning(f"⚠️  MongoDB not available: {str(e)}")
                logger.info("   Falling back to .pkl file cache...")
        
        # ========================================================================
        # PRIORITY 2: Try .pkl file cache
        # ========================================================================
        cache_file = os.path.join(os.path.dirname(__file__), "intent_embeddings_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                logger.info(f"⚡ Loading embeddings from cache ({cache_file})...")
                with open(cache_file, 'rb') as f:
                    intent_embeddings = pickle.load(f)
                
                # Validate dimensions match current provider
                if self._validate_cache_dimensions(intent_embeddings):
                    logger.info(f"✅ Embeddings loaded from cache (INSTANT - no API calls!)")
                    return intent_embeddings
                else:
                    logger.warning(
                        f"⚠️  Cache dimension mismatch detected. "
                        f"Regenerating embeddings for {EMBEDDINGS_PROVIDER}..."
                    )
                    # Delete invalid cache
                    try:
                        os.remove(cache_file)
                        logger.info(f"🗑️  Deleted invalid cache file")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not delete cache file: {e}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cache: {e}")
                logger.info("Falling back to generating embeddings...")
        else:
            logger.info("📝 No cache file found. Generating embeddings...")
            logger.info(f"   💡 TIP: Run 'python scripts/generate_intent_embeddings.py' once to create cache")
        
        # Generate embeddings (fallback)
        intent_embeddings = {}
        
        logger.info("Converting all intent examples to embeddings (using batch processing)...")
        
        for intent, examples in self.intent_examples.items():
            if EMBEDDINGS_AVAILABLE:
                try:
                    # Get embeddings for ALL examples at once (batch API call)
                    embeddings_service = get_embeddings_service()
                    embeddings = embeddings_service.embed(examples)  # Single batch call for all 20 examples
                    
                    # Convert to numpy array
                    intent_embeddings[intent] = np.array(embeddings)
                    
                    logger.debug(f"   {intent}: {len(examples)} examples embedded (batch)")
                except Exception as e:
                    logger.error(f"❌ Azure OpenAI embedding failed for {intent}: {e}")
                    logger.error("❌ CRITICAL: Cannot generate embeddings. Routing to LLM fallback.")
                    raise RuntimeError("Embedding generation failed - no cache and API unavailable") from e
            else:
                # NO mock embeddings - raise error to route to LLM
                logger.error("❌ CRITICAL: Azure embeddings not available and no cache found")
                raise RuntimeError("Embedding service unavailable - routing to LLM fallback")
        
        logger.info(f"✅ All examples embedded successfully (30 batch calls instead of 600 individual calls)")
        
        # ========================================================================
        # Save to cache for next time
        # ========================================================================
        
        # Save to MongoDB (if enabled)
        if settings.use_mongodb_for_embeddings:
            try:
                from services.mongodb_embedding_store import MongoDBEmbeddingStore
                
                logger.info(f"💾 Saving embeddings to MongoDB...")
                mongo_store = MongoDBEmbeddingStore()
                
                await mongo_store.save_embeddings(
                    intent_embeddings=intent_embeddings,
                    intent_examples=self.intent_examples,
                    embedding_provider=EMBEDDINGS_PROVIDER,
                    embedding_model="text-embedding-005" if EMBEDDINGS_PROVIDER == "Google Cloud" else "text-embedding-ada-002",
                    embedding_dimension=self._get_expected_embedding_dimension()
                )
                
                logger.info(f"✅ MongoDB cache saved!")
                
                # Auto-create vector index if it doesn't exist
                await self._ensure_vector_index(mongo_store)
                
                await mongo_store.close()
                logger.info(f"✅ MongoDB ready! Next initialization will be instant.")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save to MongoDB: {e}")
                logger.info("   Will save to .pkl file as fallback...")
        
        # Save to .pkl file ONLY if MongoDB is disabled (local dev mode)
        if not settings.use_mongodb_for_embeddings:
            try:
                logger.info(f"💾 Saving embeddings to .pkl file (local dev mode)...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(intent_embeddings, f)
                logger.info(f"✅ .pkl cache saved!")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save .pkl cache: {e}")
        else:
            logger.info(f"ℹ️  Skipping .pkl cache (MongoDB mode enabled - no local cache needed)")
        
        return intent_embeddings
    
    async def _ensure_vector_index(self, mongo_store) -> None:
        """
        Ensure vector search index exists in MongoDB Atlas.
        Creates it if it doesn't exist (required for $vectorSearch).
        """
        try:
            db = await mongo_store._get_connection()
            collection = db.intent_embeddings
            
            # Check if vector index already exists
            try:
                existing_indexes = await collection.list_search_indexes().to_list(length=None)
                for idx in existing_indexes:
                    if idx.get('name') == 'vector_index':
                        logger.info("✅ Vector search index already exists")
                        return
            except Exception as e:
                logger.debug(f"Could not list search indexes: {e}")
            
            # Create vector search index
            logger.info("🔨 Creating vector search index...")
            
            index_definition = {
                "name": "vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self._get_expected_embedding_dimension(),
                        "similarity": "cosine"
                    }]
                }
            }
            
            try:
                result = await db.command({
                    "createSearchIndexes": "intent_embeddings",
                    "indexes": [index_definition]
                })
                logger.info(f"✅ Vector search index created! (Building in background)")
                logger.info("   Note: Index may take 1-2 minutes to become queryable")
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.info("✅ Vector search index already exists")
                elif "not supported" in error_msg or "not available" in error_msg:
                    logger.warning("⚠️ Vector search not supported on this MongoDB tier (requires M10+)")
                else:
                    logger.warning(f"⚠️ Could not create vector index: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to ensure vector index: {e}")
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify user query using embedding similarity
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict with same structure as CVSIntentClassifier
            
        Raises:
            RuntimeError: If embeddings are unavailable (routes to LLM fallback)
        """
        query_lower = query.lower().strip()
        
        # Check for empty query
        if not query_lower:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False,
                'needs_clarification': False
            }
        
        # Get query embedding - FAIL if unavailable (no mock fallback)
        if EMBEDDINGS_AVAILABLE and self.embeddings_service is not None:
            try:
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"❌ Failed to get query embedding: {e}")
                raise RuntimeError("Query embedding failed - routing to LLM fallback") from e
        else:
            logger.error("❌ Embeddings service unavailable")
            raise RuntimeError("Embeddings service unavailable - routing to LLM fallback")
        
        # Calculate similarity scores for all intents
        intent_scores = {}
        
        # Check if MongoDB Vector Search is enabled
        from config.config import settings
        if settings.use_mongodb_for_embeddings:
            try:
                # Use MongoDB Atlas Vector Search (fast!)
                import asyncio
                from services.mongodb_embedding_store import MongoDBEmbeddingStore
                
                logger.debug("🔍 Using MongoDB Atlas Vector Search")
                
                # Run async vector search
                async def run_vector_search():
                    mongo_store = MongoDBEmbeddingStore()
                    try:
                        scores = await mongo_store.vector_search(
                            query_embedding=np.array(query_embedding),
                            limit=50  # Get top 50 results
                        )
                        await mongo_store.close()
                        return scores
                    except Exception as e:
                        await mongo_store.close()
                        raise e
                
                # Execute in thread pool to avoid event loop conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(run_vector_search()))
                    intent_scores = future.result(timeout=10)
                
                logger.debug(f"✅ Vector search returned {len(intent_scores)} intent scores")
                
            except Exception as e:
                logger.error(f"❌ Vector search failed: {str(e)[:100]}")
                raise RuntimeError(f"Vector search failed: {e}") from e
        
        # Python-based similarity (only when vector search is disabled)
        elif self.embeddings_service is not None:
            # Use cached Azure embeddings utility for similarity calculation
            for intent, example_embeddings in self.intent_embeddings.items():
                # Convert numpy array to list for utility function
                example_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                               for emb in example_embeddings]
                query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                
                # Calculate cosine similarity with all examples of this intent
                try:
                    similarities = self.embeddings_service.batch_similarity(query_list, example_list)
                except ValueError as e:
                    if "dimension mismatch" in str(e).lower() or "not aligned" in str(e).lower():
                        # Dimension mismatch - cache needs regeneration
                        logger.error(f"❌ {e}")
                        logger.error("❌ CRITICAL: Embedding dimension mismatch detected!")
                        logger.error("   This means the cache was created with a different embedding provider.")
                        logger.error("   Solution: Delete 'classifiers/intent_embeddings_cache.pkl' and restart server.")
                        raise RuntimeError(
                            "Embedding dimension mismatch - cache needs regeneration. "
                            "Delete 'classifiers/intent_embeddings_cache.pkl' and restart."
                        ) from e
                    else:
                        raise
                
                # Use the highest similarity as the score for this intent
                intent_scores[intent] = float(max(similarities))
        else:
            # Fallback to manual calculation if utility not available
            for intent, example_embeddings in self.intent_embeddings.items():
                similarities = self._cosine_similarity_batch(query_embedding, example_embeddings)
                intent_scores[intent] = float(np.max(similarities))
        
        # Get top intent
        if not intent_scores:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False,
                'needs_clarification': False
            }
        
        top_intent = max(intent_scores, key=intent_scores.get)
        top_score = intent_scores[top_intent]
        
        # Check if below threshold
        if top_score < self.confidence_threshold:
            return {
                'intent': 'out_of_scope',
                'confidence': top_score,
                'all_scores': intent_scores,
                'is_complex': False,
                'needs_clarification': False
            }
        
        # Classify complexity (match keyword classifier logic)
        is_complex = self._is_complex_query(query_lower, top_intent)
        
        logger.info(f"Intent: {top_intent}, Confidence: {top_score:.2f}, Complex: {is_complex}")
        
        return {
            'intent': top_intent,
            'confidence': top_score,
            'all_scores': intent_scores,
            'is_complex': is_complex,
            'needs_clarification': top_score < 0.4 and top_intent != 'out_of_scope'
        }
    
    async def classify_async(self, query: str) -> Dict[str, Any]:
        """
        Async version of classify - uses MongoDB directly like team's persistence store.
        Reuses connection via singleton factory (no asyncio.run overhead).
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict with same structure as classify()
        """
        query_lower = query.lower().strip()
        
        # Check for empty query
        if not query_lower:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False,
                'needs_clarification': False
            }
        
        # Get query embedding
        if EMBEDDINGS_AVAILABLE and self.embeddings_service is not None:
            try:
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"❌ Failed to get query embedding: {e}")
                raise RuntimeError("Query embedding failed - routing to LLM fallback") from e
        else:
            logger.error("❌ Embeddings service unavailable")
            raise RuntimeError("Embeddings service unavailable - routing to LLM fallback")
        
        # Calculate similarity scores
        intent_scores = {}
        
        from config.config import settings
        if settings.use_mongodb_for_embeddings:
            try:
                # Use MongoDB Atlas Vector Search with singleton (like team's approach)
                from services.mongodb_embedding_store import MongoDBEmbeddingStoreFactory
                
                logger.debug("🔍 Using MongoDB Atlas Vector Search (async - team pattern)")
                
                # Get singleton instance (connection reused!)
                mongo_store = MongoDBEmbeddingStoreFactory.get_instance()
                
                # Direct await - no asyncio.run() needed
                intent_scores = await mongo_store.vector_search(
                    query_embedding=np.array(query_embedding),
                    limit=50
                )
                # Don't close - singleton keeps connection for reuse
                
                logger.debug(f"✅ Vector search returned {len(intent_scores)} intent scores")
                
            except Exception as e:
                logger.error(f"❌ Vector search failed: {str(e)[:100]}")
                raise RuntimeError(f"Vector search failed: {e}") from e
        
        # Python-based similarity (fallback when MongoDB disabled)
        elif self.embeddings_service is not None:
            for intent, example_embeddings in self.intent_embeddings.items():
                example_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                               for emb in example_embeddings]
                query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                
                try:
                    similarities = self.embeddings_service.batch_similarity(query_list, example_list)
                except ValueError as e:
                    if "dimension mismatch" in str(e).lower() or "not aligned" in str(e).lower():
                        raise RuntimeError(
                            "Embedding dimension mismatch - cache needs regeneration."
                        ) from e
                    else:
                        raise
                
                intent_scores[intent] = float(max(similarities))
        else:
            for intent, example_embeddings in self.intent_embeddings.items():
                similarities = self._cosine_similarity_batch(query_embedding, example_embeddings)
                intent_scores[intent] = float(np.max(similarities))
        
        # Get top intent
        if not intent_scores:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False,
                'needs_clarification': False
            }
        
        top_intent = max(intent_scores, key=intent_scores.get)
        top_score = intent_scores[top_intent]
        
        if top_score < self.confidence_threshold:
            return {
                'intent': 'out_of_scope',
                'confidence': top_score,
                'all_scores': intent_scores,
                'is_complex': False,
                'needs_clarification': False
            }
        
        is_complex = self._is_complex_query(query_lower, top_intent)
        
        logger.info(f"Intent: {top_intent}, Confidence: {top_score:.2f}, Complex: {is_complex}")
        
        return {
            'intent': top_intent,
            'confidence': top_score,
            'all_scores': intent_scores,
            'is_complex': is_complex,
            'needs_clarification': top_score < 0.4 and top_intent != 'out_of_scope'
        }
    
    def _cosine_similarity_batch(self, query_emb: np.ndarray, example_embs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all examples
        (FALLBACK METHOD - uses utils.azure_embeddings by default)
        
        Args:
            query_emb: Query embedding (1536,)
            example_embs: Example embeddings (N, 1536)
            
        Returns:
            Array of similarities (N,)
        """
        # Normalize vectors
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        example_norms = example_embs / (np.linalg.norm(example_embs, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity
        similarities = np.dot(example_norms, query_norm)
        
        return similarities
    
    def _is_simple_query(self, query: str, intent: str) -> bool:
        """
        Determine if query is simple (can be handled with API call)
        
        Simple queries:
        - Have claim ID explicitly mentioned
        - Single intent, clear phrasing
        """
        # Has explicit claim ID
        if re.search(r'\b(CLM|claim)\s*#?\s*\w+', query, re.IGNORECASE):
            return True
        
        # Short, simple queries (< 6 words) without aggregation keywords
        words = query.split()
        if len(words) <= 6:
            complex_keywords = ['all', 'compare', 'total', 'average', 'sum', 'between', 'from', 'to']
            if not any(kw in query for kw in complex_keywords):
                return True
        
        return False
    
    def _is_complex_query(self, query: str, intent: str) -> bool:
        """
        Determine if query requires LLM reasoning
        
        Complex queries:
        - Aggregations: "total", "average", "all my claims"
        - Comparisons: "compare", "difference between"
        - Date ranges: "from January to May"
        - Multiple conditions: "and", "but", "however"
        """
        complex_patterns = [
            # Aggregations - Only match "all" when used in aggregation context
            r'\b(total|sum|average|mean)\b',  # Core aggregation words
            r'\b(all)\s+(my|the)\s+(claims|prescriptions|transactions)\b',  # "all my claims" = aggregation
            r'\b(every)\s+(claim|prescription|transaction)\b',  # "every claim" = aggregation
            r'\b(most|least|highest|lowest|expensive|cheapest)\b',
            
            # Comparisons
            r'\b(compare|comparison|difference|versus|vs)\b',
            
            # Date ranges
            r'\b(from|between|during|in)\s+\w+\s+(to|and|through)\s+\w+',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(last|past|previous)\s+(week|month|year|quarter)',
            
            # Multiple conditions
            r'\band\b.*\band\b',  # Multiple "and"s
            r'\bor\b.*\bor\b',    # Multiple "or"s
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Multi-claim queries without specific CLM ID
        if 'claims' in query and not re.search(r'CLM\d+', query, re.IGNORECASE):
            return True
        
        return False


# ================================================================================
# EMBEDDED INTENT EXAMPLES - 600 EXAMPLES (20 PER INTENT, 30 INTENTS)
# ================================================================================
# Generated from real CVS production queries using Azure OpenAI
# Original test queries EXCLUDED to avoid data leakage
# ================================================================================

CVS_INTENT_EXAMPLES = {
    
    "approval_info": [
        "Provide a detailed approval summary for this claim.",
        "Show which plan overrides were triggered during adjudication.",
        "Retrieve the transition fill status for this claim.",
        "Display the specific transition fill type applied to this claim.",
        "Tell me what plan options were active at the time of processing.",
        "Generate a report on why accumulations were not considered for this claim.",
        "Fetch the BPG configuration used to approve this claim.",
        "Give me the override details for the most recent claim.",
        "Provide information on any member PA or Smart PA applied.",
        "Show the logic behind the approval decision for this claim.",
        "Display all plan configuration overrides that affected this claim.",
        "Retrieve the adjudication pathway for this claim.",
        "Generate a TF summary for this claim.",
        "Show transition fill details for this claim.",
        "Provide TF status for this claim.",
        "Display transition fill approval information.",
        "Fetch TF configuration for this claim.",
        "Tell me the TF eligibility for this member.",
        "Give me transition fill override details.",
        "Retrieve the TF type applied to this claim.",
    ],

    "audit_info": [
        "Generate the audit log for this claim",
        "Show me the change history for this claim",
        "Display the audit information for this claim",
        "Give me the modification record for this claim",
        "Tell me the audit details for this claim",
        "Provide the full audit report for this claim",
        "Retrieve the change log for this claim",
        "Fetch the update history for this claim",
        "Generate the modification audit for this claim",
        "Show the claim history for this claim",
        "Display the audit trail for this claim",
        "Give me the edit history for this claim",
        "Tell me the claim modification details",
        "Provide the audit summary for this claim",
        "Retrieve the claim audit for this claim",
        "Fetch the claim modification log",
        "Generate the history of changes for this claim",
        "Show me the audit record for this claim",
        "Display the modification details for this claim",
        "Give me the audit information for this claim",
    ],

    "beneficiary_info": [
        "Generate the current benefit phase for this claim.",
        "Show the member's coverage details associated with this claim.",
        "Display the accumulation status for the member on this claim.",
        "Give me the member's benefit type for this claim.",
        "Tell me if medical dollars are included in the member's accumulation.",
        "Provide the linked LOE information for the member on this claim.",
        "Retrieve the member's coverage tier for this claim.",
        "Fetch the benefit phase the member is currently in for this claim.",
        "Generate the member's eligibility information for this claim.",
        "Show the member's plan details for this claim.",
        "Display whether medical expenses contribute to the accumulation.",
        "Give me the member's insurance coverage type.",
        "Tell me the current benefit phase for the member on this claim.",
        "Provide the linked member levels of evidence (LOE).",
        "Retrieve the member's benefit eligibility.",
        "Fetch the member's coverage information for this prescription.",
        "Generate a summary of the member's benefit phase.",
        "Show the member's accumulation rules for this claim.",
        "Display the member's coverage classification.",
        "Give me the member's benefit plan configuration.",
    ],

    "claim_status": [
        "Generate a full claim summary for this claim.",
        "Show the current status of this claim.",
        "Display all details for this claim.",
        "Give me the processing status for this claim.",
        "Tell me the progress of this claim.",
        "Provide the adjudication outcome for this claim.",
        "Retrieve the complete claim record.",
        "Fetch the claim details for this claim.",
        "Generate a breakdown of this claim.",
        "Show the status update for this claim.",
        "Give me a claim summary for this claim.",
        "Generate a claim summary.",
        "Show claim summary for this claim.",
        "Provide a summary of this claim.",
        "What happened on claim sequences?",
        "Compare this claim to another claim.",
        "Tell me about this claim.",
        "Show the full record for this claim.",
        "Display the adjudication details for this claim.",
        "Give me the comprehensive summary.",
    ],

    "cob_info": [
        "Generate a COB summary for this claim.",
        "Show the coordination of benefits details for this claim.",
        "Display the other insurance information for this claim.",
        "Give me the COB pricing breakdown for this claim.",
        "Tell me how other insurance affected this claim.",
        "Provide the secondary insurance details for this claim.",
        "Retrieve the COB calculation for this claim.",
        "Fetch the coordination details for this claim.",
        "Generate the COB pricing information.",
        "Show how coordination of benefits was applied to this claim.",
        "Display the other payer information for this claim.",
        "Give me the dual coverage details for this claim.",
        "Tell me the COB setup for this claim.",
        "Provide the secondary payer information.",
        "Retrieve the coordination of benefits report.",
        "Fetch the other insurance impact on this claim.",
        "Generate the COB adjudication details.",
        "Show the primary and secondary insurance breakdown.",
        "Display the COB configuration for this claim.",
        "Give me the coordination pricing summary.",
    ],

    "compound_info": [
        "Generate the compound medication details for this claim.",
        "Show the ingredient breakdown for this claim.",
        "Display the MIC information for this claim.",
        "Give me the compound ingredient costs for this claim.",
        "Tell me if this claim is for a compounded medication.",
        "Provide the funded and unfunded costs for the compound in this claim.",
        "Retrieve the ingredient list for this compound claim.",
        "Fetch the compound drug details for this claim.",
        "Generate the MIC cost breakdown.",
        "Show the individual ingredient pricing.",
        "Display whether this is a compound prescription.",
        "Give me the compound formulation details.",
        "Tell me the ingredient costs for the compounded medication.",
        "Provide the compound medication summary.",
        "Retrieve the MIC details and ingredient breakdown.",
        "Fetch the compound pricing information.",
        "Generate the ingredient cost report.",
        "Show the funded versus unfunded ingredient costs.",
        "Display the compound claim details.",
        "Give me the MIC summary for this prescription.",
    ],

    "date_range_claims": [
        "Generate a list of claims that contributed to the deductible.",
        "Show all claims for this drug in the member's history.",
        "Display claims from the past six months.",
        "Give me the claims that affected the out-of-pocket maximum.",
        "Tell me which claims contributed to the member's accumulation.",
        "Provide a history of all claims for this medication.",
        "Retrieve claims within a specific date range.",
        "Fetch the member's claim history for this drug.",
        "Generate a report of claims affecting deductible and OOP.",
        "Show the claims that impacted the member's benefit phase.",
        "Display all prescription claims for this member.",
        "Give me the list of claims for this year.",
        "Tell me the claims that contributed to cost-sharing.",
        "Provide the claim history for this member and drug.",
        "Retrieve all claims that affected the accumulation.",
        "Fetch the deductible-contributing claims.",
        "Generate a summary of claims over the last quarter.",
        "Show the member's prescription history.",
        "Display the claims from January to March.",
        "Give me all claims for this drug type.",
    ],

    "daw_info": [
        "Generate the DAW status for this claim.",
        "Show the dispense as written details for this claim.",
        "Display whether brand name was required for this claim.",
        "Give me the DAW code for this prescription.",
        "Tell me if generic substitution was allowed for this claim.",
        "Provide the brand versus generic information for this claim.",
        "Retrieve the DAW indicator for this claim.",
        "Fetch the dispense as written status.",
        "Generate the substitution details for this prescription.",
        "Show whether brand was medically necessary.",
        "Display the DAW selection for this claim.",
        "Give me the brand name requirement details.",
        "Tell me if a generic could be substituted.",
        "Provide the DAW configuration for this claim.",
        "Retrieve the dispense as written code.",
        "Fetch the brand versus generic status.",
        "Generate the substitution rules for this claim.",
        "Show the DAW designation.",
        "Display the brand requirement information.",
        "Give me the generic availability status.",
    ],

    "drug_info": [
        "Generate the drug status for this claim.",
        "Show the medication details for this claim.",
        "Display the drug information for this prescription.",
        "Give me the formulary status for the drug in this claim.",
        "Tell me which drug was dispensed for this claim.",
        "Provide the medication classification for this claim.",
        "Retrieve the drug setup used in RxClaim for this claim.",
        "Fetch the drug name and NDC for this prescription.",
        "Generate the drug category details.",
        "Show the formulary position for this medication.",
        "Display the drug type and classification.",
        "Give me the GPI and therapeutic class.",
        "Tell me the drug status assigned by RxClaim.",
        "Provide the medication tier information.",
        "Retrieve the drug details for this claim.",
        "Fetch the drug configuration from the claim.",
        "Generate the formulary report for this drug.",
        "Show the drug classification details.",
        "Display the medication status.",
        "Give me the drug tier and formulary placement.",
    ],

    "drug_interaction_info": [
        "Generate the DUR edits applied to this claim.",
        "Show the drug utilization review details for this claim.",
        "Display the DUR outcomes for this prescription.",
        "Give me the interaction alerts for this claim.",
        "Tell me which DUR edits were triggered for this claim.",
        "Provide the drug interaction warnings.",
        "Retrieve the DUR edit results for this claim.",
        "Fetch the utilization review information.",
        "Generate the DUR report for this claim.",
        "Show the drug interaction outcomes.",
        "Display the clinical edits that were triggered.",
        "Give me the DUR override details.",
        "Tell me the utilization review results.",
        "Provide the interaction screening outcomes.",
        "Retrieve the DUR edit codes for this claim.",
        "Fetch the drug utilization details.",
        "Generate the interaction alert summary.",
        "Show the clinical screening results.",
        "Display the DUR processing details.",
        "Give me the drug interaction report.",
    ],

    "fill_date_info": [
        "Generate the fill date for this claim.",
        "Show when the prescription was filled for this claim.",
        "Display the dispense date for this claim.",
        "Give me the service date for this claim.",
        "Tell me when the medication was dispensed.",
        "Provide the date the prescription was filled.",
        "Retrieve the fill date details for this claim.",
        "Fetch the dispensing date for this prescription.",
        "Generate the date of service for this claim.",
        "Show the fill date information.",
        "Display when the pharmacy filled this prescription.",
        "Give me the date the medication was picked up.",
        "Tell me the dispensing date for this claim.",
        "Provide the fill date and time.",
        "Retrieve the service date for this prescription.",
        "Fetch the date when this was filled.",
        "Generate the dispensing date report.",
        "Show the fill timestamp.",
        "Display the prescription fill date.",
        "Give me the date of fill for this claim.",
    ],

    "generic_availability": [
        "Generate a list of generic alternatives for this claim.",
        "Show the substitute medications available.",
        "Display the generic options for this drug.",
        "Give me the alternative medications for this claim.",
        "Tell me what generic drugs are available.",
        "Provide the therapeutic alternatives.",
        "Retrieve the substitute drug list.",
        "Fetch the generic availability information.",
        "Generate the alternative medication options.",
        "Show the formulary alternatives for this drug.",
        "Display the generic substitutes.",
        "Give me the list of cheaper drug options.",
        "Tell me the available generic medications.",
        "Provide the therapeutic equivalent drugs.",
        "Retrieve the substitute options for this prescription.",
        "Fetch the generic alternative details.",
        "Generate the drug substitution recommendations.",
        "Show the formulary-approved alternatives.",
        "Display the generic medication options.",
        "Give me the substitute drug information.",
    ],

    "government_claim_type": [
        "Generate the government claim type for this claim.",
        "Show whether this is a Medicare or Medicaid claim.",
        "Display the government program classification for this claim.",
        "Give me the claim type designation.",
        "Tell me if this claim is a government claim.",
        "Provide the Medicare or Medicaid status.",
        "Retrieve the government program details.",
        "Fetch the claim classification for this claim.",
        "Generate the program type information.",
        "Show the government claim designation.",
        "Display whether this is a public insurance claim.",
        "Give me the Medicare type for this claim.",
        "Tell me the government program classification.",
        "Provide the claim type (Medicare Part D, Medicaid, etc.).",
        "Retrieve the government claim category.",
        "Fetch the public program details.",
        "Generate the claim type report.",
        "Show the government insurance classification.",
        "Display the Medicare or Medicaid designation.",
        "Give me the program type for this claim.",
    ],

    "greeting": [
        "Generate a welcome message",
        "Show me the greeting",
        "Display hello",
        "Give me a hi",
        "Tell me hello there",
        "Provide greetings",
        "Retrieve the welcome",
        "Fetch a greeting",
        "Generate hello",
        "Show a welcome",
        "Display good morning",
        "Give me a good afternoon",
        "Tell me good evening",
        "Provide a hi",
        "Retrieve greetings",
        "Fetch hello",
        "Generate a hi there",
        "Show good day",
        "Display howdy",
        "Give me a hey",
    ],

    "help": [
        "Generate instructions on how to submit a claim properly.",
        "Show me the steps to avoid claim rejection.",
        "Display the correct submission process.",
        "Give me guidance on filing claims.",
        "Tell me how to submit claims without errors.",
        "Provide the proper claim submission procedure.",
        "Retrieve the instructions for correct claim filing.",
        "Fetch the help documentation for claim submission.",
        "Generate the guidelines for avoiding rejections.",
        "Show the best practices for claim submission.",
        "Display the claim submission checklist.",
        "Give me the process to follow for successful claims.",
        "Tell me the requirements for claim acceptance.",
        "Provide the claim submission help.",
        "Retrieve the instructions to prevent denials.",
        "Fetch the submission guidelines.",
        "Generate the correct filing procedures.",
        "Show the claim submission protocol.",
        "Display the help for proper claim filing.",
        "Give me the instructions for avoiding errors.",
    ],

    "mail_order_info": [
        "Generate the mail order status for this claim.",
        "Show whether this was a home delivery prescription.",
        "Display the delivery method for this claim.",
        "Give me the mail order details for this claim.",
        "Tell me if this prescription was shipped.",
        "Provide the home delivery information.",
        "Retrieve the mail order status.",
        "Fetch the delivery type for this prescription.",
        "Generate the shipping details for this claim.",
        "Show whether this was a mail order claim.",
        "Display the delivery method used.",
        "Give me the home delivery status.",
        "Tell me if this was sent by mail.",
        "Provide the mail order designation.",
        "Retrieve the shipping information.",
        "Fetch the delivery details for this claim.",
        "Generate the mail order report.",
        "Show the home delivery status.",
        "Display whether this was mailed.",
        "Give me the delivery method information.",
    ],

    "medicare_part_d": [
        "Generate a Medicare Part D summary for this claim.",
        "Show the PDE details for this claim.",
        "Display the MEDD pricing including LICS and N1s.",
        "Give me the Part D information for this claim.",
        "Tell me the Medicare coverage details for this claim.",
        "Provide the PDE report for this prescription.",
        "Retrieve the Part D summary.",
        "Fetch the Medicare pricing details.",
        "Generate the LICS and N1 breakdown for this claim.",
        "Show the Part D plan information.",
        "Display the PDE submission details.",
        "Give me the Medicare Part D coverage.",
        "Tell me the MEDD pricing summary.",
        "Provide the Part D benefit details.",
        "Retrieve the PDE information.",
        "Fetch the Medicare coverage report.",
        "Generate the Part D pricing breakdown.",
        "Show the LICS details for this claim.",
        "Display the N1 and MEDD information.",
        "Give me the Medicare Part D summary.",
    ],

    "multi_claim_summary": [
        "Generate a summary of ALL claims for this member",
        "Display the list of MULTIPLE claims associated with this member",
        "Retrieve ALL claim records in the system",
        "Provide details for EVERY claim linked to this member",
        "Show the complete set of ALL claims",
        "Give me a summary of MULTIPLE claims",
        "Tell me about ALL claims under this member",
        "Fetch EVERY claim related to this member",
        "Generate a report of ALL claims in the database",
        "Display ALL available claims for this member",
        "Retrieve ALL claims in the system",
        "Provide a summary of MULTIPLE claims in history",
        "Show the FULL LIST of claims",
        "Give me details for ALL claims",
        "Tell me about MULTIPLE claims",
        "Fetch ALL claim records",
        "List EVERY claim in the database",
        "Display MULTIPLE claims",
        "Retrieve the complete set of ALL claims",
        "Show me ALL claims for this member",
    ],

    "network_info": [
        "Generate the pharmacy network details for this claim.",
        "Show which network was used for this claim.",
        "Display the network information for this prescription.",
        "Give me the pharmacy chain details for this claim.",
        "Tell me which network processed this claim.",
        "Provide the pharmacy network classification.",
        "Retrieve the network type for this claim.",
        "Fetch the pharmacy network information.",
        "Generate the network details used for payment.",
        "Show the pharmacy chain network.",
        "What pharmacy network did the claim pay with?",
        "Which network was used to pay this claim?",
        "Tell me the payment network for this claim.",
        "Show the network that paid for this claim.",
        "What network processed the payment for this claim?",
        "Display the paying network information.",
        "Which pharmacy network handled the payment?",
        "Retrieve the network used for claim payment.",
        "Give me the payment network details.",
        "Show the network that reimbursed this claim.",
    ],

    "out_of_scope": [
        # Clearly off-topic questions (no trigger words like Provide/Show/Display)
        "What's the weather forecast?",
        "I need a pizza recipe",
        "Book me a flight to Hawaii",
        "Sports scores from last night",
        "Latest political news",
        "Best restaurants near me",
        "Stock market performance today",
        "Movie recommendations for tonight",
        "Celebrity gossip updates",
        "Travel tips for Europe",
        "Historical facts about Rome",
        "Science experiments for kids",
        # Question format (interrogative)
        "Why is the sky blue?",
        "What is the weather like today?",
        "How do I cook pasta?",
        "When is the Super Bowl?",
        "Who won the election?",
        "Where can I find good restaurants?",
        "What time does the movie start?",
        "How does gravity work?",
        "Why do birds fly south?",
        "What's the capital of France?",
        # General off-topic
        "Random trivia question",
        "Joke of the day",
        "Music playlist suggestions",
        "Fitness workout routines",
        "Gaming tips and tricks",
        "Fashion advice for summer",
        # Lorem ipsum / Junk text / Gibberish
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        "Quisque faucibus ex sapien vitae pellentesque sem placerat",
        "Tempus leo eu aenean sed diam urna tempor pulvinar",
        "Vivamus fringilla lacus nec metus bibendum egestas",
        "Ad litora torquent per conubia nostra inceptos himenaeos",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem",
        "Nemo enim ipsam voluptatem quia voluptas sit aspernatur",
        "Neque porro quisquam est qui dolorem ipsum quia dolor",
        "asdfghjkl qwertyuiop zxcvbnm",
        "abcdefghijklmnopqrstuvwxyz random letters",
        "test test test test test test",
        "blah blah blah blah blah",
        "foo bar baz qux quux corge",
        "hello hello hello hello",
        "aaaaaa bbbbbb cccccc dddddd",
        "12345 67890 abcde fghij",
    ],

    "pharmacy_info": [
        "Generate the pharmacy details for this claim.",
        "Show where the prescription was filled.",
        "Display the dispensing pharmacy for this claim.",
        "Give me the pharmacy name where this claim was filled.",
        "Tell me which pharmacy dispensed this prescription.",
        "Provide the pharmacy location information.",
        "Retrieve the dispensing pharmacy details.",
        "Fetch the pharmacy information for this claim.",
        "Generate the store location where this was filled.",
        "Show the pharmacy name and address.",
        "Display the dispensing location.",
        "Give me the CVS store details.",
        "Tell me where the member got this prescription.",
        "Provide the pharmacy NCPDP number.",
        "Retrieve the dispensing pharmacy location.",
        "Fetch the store information.",
        "Generate the pharmacy report.",
        "Show where the medication was dispensed.",
        "Display the pharmacy details.",
        "Give me the filling pharmacy information.",
    ],

    "prescriber_info": [
        "Generate the prescriber details for this claim.",
        "Show who prescribed the medication for this claim.",
        "Display the physician information for this prescription.",
        "Give me the doctor's name for this claim.",
        "Tell me which physician wrote this prescription.",
        "Provide the prescriber NPI for this claim.",
        "Retrieve the ordering provider details.",
        "Fetch the prescriber information.",
        "Generate the physician report for this claim.",
        "Show the doctor's contact information.",
        "Display the prescriber's name and credentials.",
        "Give me the provider details.",
        "Tell me who ordered this medication.",
        "Provide the prescribing physician information.",
        "Retrieve the prescriber NPI and name.",
        "Fetch the doctor's details.",
        "Generate the provider summary.",
        "Show the prescribing physician.",
        "Display the ordering provider information.",
        "Give me the prescriber name.",
    ],

    "pricing_info": [
        "Show the detailed pricing breakdown for this claim.",
        "Provide the copay calculation steps for this claim.",
        "Retrieve the ingredient cost and associated fees for this claim.",
        "Display the manufacturer rebate applied to this claim.",
        "Give me the pricing schedule used for this claim.",
        "Fetch the patient pay details for this claim.",
        "Tell me the final out-of-pocket amount for this claim.",
        "Generate a summary of all pricing components for this claim.",
        "Show the co-pay modifier and its impact on this claim.",
        "Provide the total cost and copay for this claim.",
        "Generate a pricing summary for this claim.",
        "Show pricing summary for this claim.",
        "Display the pricing breakdown.",
        "What is the cost of this claim?",
        "How much did I pay?",
        "Tell me the pricing details.",
        "What's the total cost of my prescriptions?",
        "Provide the pricing summary for the most recent claim.",
        "Retrieve the final copay amount for the last processed claim.",
        "Display the pricing information for this claim.",
    ],

    "prior_auth_info": [
        "Generate the prior authorization details for this claim.",
        "Show the PA status for this claim.",
        "Display the Smart PA information for this claim.",
        "Give me the Member PA details for this claim.",
        "Tell me if prior authorization was required.",
        "Provide the PA number and approval status.",
        "Retrieve the prior auth details for this prescription.",
        "Fetch the authorization information.",
        "Generate the PA summary for this claim.",
        "Show whether PA was approved or denied.",
        "Display the authorization status.",
        "Give me the Smart PA configuration.",
        "Tell me the prior auth requirements.",
        "Provide the Member PA summary.",
        "Retrieve the PA approval details.",
        "Fetch the authorization status for this drug.",
        "Generate the prior authorization report.",
        "Show the PA type and approval.",
        "Display the authorization details.",
        "Give me the prior auth information.",
    ],

    "reimbursement_info": [
        "Generate the reimbursement details for this claim.",
        "Show what was reimbursed for the paper claim.",
        "Display the payment information for this claim.",
        "Give me the reimbursement amount for this claim.",
        "Tell me what was paid for this claim.",
        "Provide the reimbursement breakdown.",
        "Retrieve the payment details.",
        "Fetch the reimbursement information.",
        "Generate the payment summary.",
        "Show the reimbursed amount.",
        "Display the payment calculation.",
        "Give me the reimbursement rationale.",
        "Tell me why this amount was reimbursed.",
        "Provide the payment details for the paper claim.",
        "Retrieve the reimbursement report.",
        "Fetch the payment information.",
        "Generate the reimbursement summary.",
        "Show what the pharmacy was paid.",
        "Display the reimbursement calculation.",
        "Give me the payment breakdown.",
    ],

    "rejection_reasons": [
        "Generate the specific rejection reasons for this claim.",
        "Show the edits that caused this claim to reject.",
        "Display the rejection details for this claim.",
        "Give me the list of rejection codes for this claim.",
        "Tell me why this claim was denied.",
        "Provide the rejection explanation for this claim.",
        "Retrieve the failed edits for this claim.",
        "Fetch the reason codes for the rejection of this claim.",
        "Generate the rejection rationale for this claim.",
        "Show the specific edits that led to the claim rejection.",
        "Display the denial reasons for the current claim.",
        "Give me the claim rejection breakdown.",
        "Tell me which edits triggered the rejection for this claim.",
        "Provide the details on why the claim was not accepted.",
        "Retrieve the rejection information for this claim.",
        "Fetch the failed edit codes for the claim.",
        "Generate a summary of the rejection reasons for the claim.",
        "Show the claim rejection cause.",
        "Display the edits that resulted in the claim denial.",
        "Give me the technical reason for the claim rejection.",
        "What can be done to overcome the rejection?",
        "How to resolve a rejected claim?",
        "Steps to overturn a rejection for this claim.",
        "Options to fix a denied claim.",
    ],

    "reversal_info": [
        "Generate the reversal details for this claim.",
        "Show any adjustments made to this claim.",
        "Display the R&R information for this claim.",
        "Give me the manual adjustment details.",
        "Tell me if this claim was reversed.",
        "Provide the modification history.",
        "Retrieve the reversal and resubmission details.",
        "Fetch the adjustment information.",
        "Generate the R&R report for this claim.",
        "Show any claim modifications.",
        "Display the reversal status.",
        "Give me the adjustment details.",
        "Tell me about any manual changes.",
        "Provide the reversal information.",
        "Retrieve the modification details.",
        "Fetch the R&R status.",
        "Generate the adjustment summary.",
        "Show the claim reversal details.",
        "Display any resubmissions.",
        "Give me the reversal report.",
    ],

    "rx_details": [
        "Generate the RX number for this claim.",
        "Show the prescription details for this claim.",
        "Display the quantity dispensed for this claim.",
        "Give me the days supply for this claim.",
        "Tell me the fill number for this prescription.",
        "Provide the RX details for this claim.",
        "Retrieve the prescription number.",
        "Fetch the quantity and days supply.",
        "Generate the RX information.",
        "Show the prescription fill details.",
        "Display the RX number and fill count.",
        "Give me the dispensing quantity.",
        "Tell me the days supply for this prescription.",
        "Provide the fill number information.",
        "Retrieve the RX details.",
        "Fetch the prescription strength and quantity.",
        "Generate the prescription summary.",
        "Show the RX number.",
        "Display the fill details.",
        "Give me the prescription information.",
    ],

    "settlement_info": [
        "Generate the settlement codes for this claim.",
        "Show the pharmacy response for this claim.",
        "Display the settlement information sent to the pharmacy.",
        "Give me the feedback codes for this claim.",
        "Tell me what was sent back to the pharmacy.",
        "Provide the settlement details.",
        "Retrieve the pharmacy response codes.",
        "Fetch the settlement information.",
        "Generate the settlement report.",
        "Show the response sent to the pharmacy.",
        "Display the settlement codes.",
        "Give me the pharmacy feedback.",
        "Tell me the settlement status.",
        "Provide the response codes.",
        "Retrieve the settlement details.",
        "Fetch the pharmacy response.",
        "Generate the settlement summary.",
        "Show the codes sent to the pharmacy.",
        "Display the settlement feedback.",
        "Give me the response information.",
    ],
}

