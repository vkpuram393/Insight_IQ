#!/usr/bin/env python3
"""
Generate Intent Embeddings to MongoDB

This script generates embeddings for all intent examples and saves them to MongoDB.
Uses nested structure: one document per intent with embedded examples array.

Usage:
    python scripts/generate_embeddings_to_mongodb.py
    python scripts/generate_embeddings_to_mongodb.py --force  # Force regeneration

Requirements:
- MongoDB running (locally or remote)
- GCP credentials configured (for Google embeddings)
- use_google_embeddings=True in config (or Azure credentials if False)
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.mongodb_embedding_store import MongoDBEmbeddingStore
from config.config import settings
from core.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# Import embeddings service based on configuration
if settings.use_google_embeddings:
    from services.google_embeddings import get_embedding, get_google_embeddings as get_embeddings_service
    EMBEDDINGS_PROVIDER = "Google Cloud Vertex AI"  # Must match classifier
    EMBEDDINGS_MODEL = "text-embedding-005"
else:
    from services.azure_embeddings import get_embedding, get_azure_embeddings as get_embeddings_service
    EMBEDDINGS_PROVIDER = "Azure OpenAI"
    EMBEDDINGS_MODEL = "text-embedding-ada-002"


def load_intent_examples() -> dict:
    """Load intent examples from embedded classifier"""
    from classifiers.embedded_classifier import CVSIntentEmbedded
    
    # Create temporary instance just to get examples
    temp_classifier = CVSIntentEmbedded.__new__(CVSIntentEmbedded)
    temp_classifier.intent_examples = temp_classifier._build_intent_examples()
    
    return temp_classifier.intent_examples


async def generate_embeddings_to_mongodb(force: bool = False):
    """Generate embeddings and save to MongoDB"""
    
    print("=" * 80)
    print("📊 MongoDB Intent Embeddings Generator")
    print("=" * 80)
    print()
    
    print(f"🔧 Configuration:")
    print(f"   Provider: {EMBEDDINGS_PROVIDER}")
    print(f"   Model: {EMBEDDINGS_MODEL}")
    print(f"   Database: {settings.mongodb_database_name}")
    print(f"   Connection: {settings.mongodb_connection_string}")
    print()
    
    # Initialize MongoDB store
    mongo_store = MongoDBEmbeddingStore()
    
    try:
        # Load intent examples
        print("📝 Loading intent examples...")
        intent_examples = load_intent_examples()
        print(f"   ✅ Loaded {len(intent_examples)} intents")
        total_examples = sum(len(examples) for examples in intent_examples.values())
        print(f"   ✅ Total examples: {total_examples}")
        print()
        
        # Get expected dimension
        try:
            test_embedding = get_embedding("test")
            embedding_dimension = len(test_embedding)
            print(f"📏 Embedding dimension: {embedding_dimension}")
            print()
        except Exception as e:
            print(f"❌ Failed to get test embedding: {e}")
            print("   Make sure GCP credentials are configured!")
            return
        
        # Check if cache exists and is valid
        if not force:
            print("🔍 Checking existing cache...")
            is_valid = await mongo_store.check_cache_validity(
                current_provider=EMBEDDINGS_PROVIDER,
                current_dimension=embedding_dimension,
                current_examples=intent_examples
            )
            
            if is_valid:
                print()
                print("=" * 80)
                print("✅ CACHE IS VALID - NO REGENERATION NEEDED")
                print("=" * 80)
                print()
                print("MongoDB already has up-to-date embeddings!")
                print("Use --force flag to regenerate anyway.")
                print()
                return
            else:
                print("   ⚠️  Cache invalid or outdated - regeneration needed")
                print()
        else:
            print("🔄 Force regeneration requested")
            print()
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        print(f"   This will make ~{len(intent_examples)} API calls (batch processing)")
        print()
        
        embeddings_service = get_embeddings_service()
        intent_embeddings = {}
        
        for idx, (intent, examples) in enumerate(intent_examples.items(), 1):
            print(f"   [{idx}/{len(intent_examples)}] {intent}: {len(examples)} examples...", end=" ")
            
            try:
                # Get embeddings for all examples at once (batch API call)
                embeddings = embeddings_service.embed(examples)
                
                # Convert to numpy array
                intent_embeddings[intent] = np.array(embeddings, dtype=np.float32)
                
                print("✅")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Failed to generate embeddings for {intent}: {e}")
                raise
        
        print()
        print(f"✅ All embeddings generated successfully!")
        print()
        
        # Save to MongoDB
        print("💾 Saving to MongoDB...")
        await mongo_store.save_embeddings(
            intent_embeddings=intent_embeddings,
            intent_examples=intent_examples,
            embedding_provider=EMBEDDINGS_PROVIDER,
            embedding_model=EMBEDDINGS_MODEL,
            embedding_dimension=embedding_dimension
        )
        
        print()
        print("=" * 80)
        print("🎉 SUCCESS!")
        print("=" * 80)
        print()
        print(f"✅ Saved {len(intent_embeddings)} intents ({total_examples} examples) to MongoDB")
        print(f"✅ Database: {settings.mongodb_database_name}")
        print(f"✅ Collections: intent_embeddings (30 docs) + embedding_metadata (1 doc)")
        print()
        print("Next steps:")
        print("  1. Restart your server")
        print("  2. Embeddings will load from MongoDB instantly!")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print()
        print(f"Failed: {str(e)}")
        print()
        logger.error(f"Failed to generate embeddings to MongoDB: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        await mongo_store.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate intent embeddings and save to MongoDB"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache is valid"
    )
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(generate_embeddings_to_mongodb(force=args.force))


if __name__ == "__main__":
    main()

