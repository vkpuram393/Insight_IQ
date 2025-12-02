#!/usr/bin/env python3
"""
Setup MongoDB Atlas Vector Search Index

This script creates a vector search index on the intent_embeddings collection
to enable fast similarity search using MongoDB's native vector search capabilities.

Requirements:
- MongoDB Atlas M10+ cluster (vector search requires Atlas)
- Motor library installed
- MongoDB credentials in environment variables or .env file

Usage:
    python scripts/setup_vector_index.py
"""
import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
from config.config import settings


async def create_vector_index():
    """Create vector search index on intent_embeddings collection"""
    
    print("=" * 80)
    print("🔧 MongoDB Atlas Vector Search Index Setup")
    print("=" * 80)
    print()
    
    # Connect to MongoDB
    print(f"📡 Connecting to MongoDB...")
    print(f"   Database: {settings.mongodb_database_name}")
    print(f"   Collection: intent_embeddings")
    print()
    
    try:
        client = AsyncIOMotorClient(
            settings.mongodb_connection_string,
            serverSelectionTimeoutMS=10000,
            tlsCAFile=certifi.where()
        )
        
        # Test connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        print()
        
        db = client[settings.mongodb_database_name]
        collection = db.intent_embeddings
        
        # Check if index already exists
        print("🔍 Checking for existing vector search indexes...")
        
        try:
            # List existing search indexes
            existing_indexes = await collection.list_search_indexes().to_list(length=None)
            
            vector_index_exists = False
            for idx in existing_indexes:
                if idx.get('name') == 'vector_index':
                    print(f"⚠️  Vector index 'vector_index' already exists")
                    print(f"   Status: {idx.get('status', 'unknown')}")
                    vector_index_exists = True
                    break
            
            if vector_index_exists:
                print()
                response = input("Do you want to recreate the index? (yes/no): ")
                if response.lower() != 'yes':
                    print("❌ Cancelled. Keeping existing index.")
                    client.close()
                    return
                
                # Drop existing index
                print("🗑️  Dropping existing vector index...")
                await collection.drop_search_index('vector_index')
                print("✅ Existing index dropped")
                print()
                
                # Wait for deletion to complete
                print("⏳ Waiting for index deletion to complete...")
                await asyncio.sleep(5)
        
        except Exception as e:
            # list_search_indexes might not be available on all versions
            print(f"ℹ️  Could not list existing indexes (this is okay): {str(e)[:100]}")
            print()
        
        # Create vector search index
        print("🔨 Creating vector search index...")
        print()
        print("Index Configuration:")
        print("  - Name: vector_index")
        print("  - Type: vectorSearch")
        print("  - Field: embedding")
        print("  - Dimensions: 768 (Google text-embedding-005)")
        print("  - Similarity: cosine")
        print()
        
        # Create the index using createSearchIndexes command
        index_definition = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine"
                }]
            }
        }
        
        try:
            result = await db.command({
                "createSearchIndexes": "intent_embeddings",
                "indexes": [index_definition]
            })
            
            print("✅ Vector search index created successfully!")
            print(f"   Result: {result}")
            print()
            
            print("⏳ Index is being built in the background...")
            print("   This may take 1-2 minutes for 594 embeddings")
            print()
            
            # Wait and check index status
            print("🔍 Monitoring index build status...")
            for i in range(12):  # Check for up to 2 minutes
                await asyncio.sleep(10)
                
                try:
                    indexes = await collection.list_search_indexes().to_list(length=None)
                    for idx in indexes:
                        if idx.get('name') == 'vector_index':
                            status = idx.get('status', 'unknown')
                            queryable = idx.get('queryable', False)
                            
                            print(f"   [{i*10}s] Status: {status} | Queryable: {queryable}")
                            
                            if queryable:
                                print()
                                print("🎉 Index is ready and queryable!")
                                break
                    else:
                        continue
                    break
                except Exception as e:
                    print(f"   Could not check status: {str(e)[:80]}")
            
            print()
            print("=" * 80)
            print("✅ SETUP COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("  1. Run: python scripts/migrate_to_flat_structure.py")
            print("  2. Test: python scripts/test_vector_search.py")
            print()
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for common errors
            if "search index not supported" in error_msg.lower():
                print()
                print("❌ ERROR: Vector Search is not available on your cluster tier")
                print()
                print("Vector Search requires:")
                print("  - MongoDB Atlas (not self-hosted MongoDB)")
                print("  - M10+ cluster tier (M0/M2/M5 free tiers don't support it)")
                print()
                print("Current cluster appears to be a lower tier or self-hosted.")
                print()
                print("Options:")
                print("  1. Upgrade to Atlas M10+ cluster")
                print("  2. Continue using current Python-based similarity search")
                print()
                sys.exit(1)
            
            elif "createSearchIndexes" in error_msg:
                print()
                print("❌ ERROR: createSearchIndexes command not available")
                print()
                print("This might be because:")
                print("  - MongoDB version is too old (need 6.0.10+ or 7.0+)")
                print("  - Atlas Search is not enabled on your cluster")
                print()
                print(f"Error details: {error_msg}")
                print()
                sys.exit(1)
            
            else:
                print()
                print(f"❌ ERROR: {error_msg}")
                print()
                sys.exit(1)
        
        client.close()
        
    except Exception as e:
        print()
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        print()
        print("Please check:")
        print("  - MongoDB connection string is correct")
        print("  - Network access is configured in Atlas")
        print("  - Database credentials are valid")
        print()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(create_vector_index())
    except KeyboardInterrupt:
        print()
        print("❌ Cancelled by user")
        sys.exit(1)

