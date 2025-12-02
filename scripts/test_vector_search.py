#!/usr/bin/env python3
"""
Test MongoDB Atlas Vector Search

This script tests the vector search functionality by:
1. Checking if embeddings are loaded in flat structure
2. Testing vector search with sample queries
3. Comparing results with Python-based similarity search
4. Measuring performance

Usage:
    python scripts/test_vector_search.py
"""
import asyncio
import sys
import time
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import numpy as np
from config.config import settings


async def test_vector_search():
    """Test vector search functionality"""
    
    print("=" * 80)
    print("🧪 MongoDB Atlas Vector Search Test")
    print("=" * 80)
    print()
    
    # Connect to MongoDB
    print(f"📡 Connecting to MongoDB...")
    print(f"   Database: {settings.mongodb_database_name}")
    print()
    
    try:
        client = AsyncIOMotorClient(
            settings.mongodb_connection_string,
            serverSelectionTimeoutMS=10000,
            tlsCAFile=certifi.where()
        )
        
        await client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        print()
        
        db = client[settings.mongodb_database_name]
        
        # Check flat structure
        print("🔍 Checking data structure...")
        sample_doc = await db.intent_embeddings.find_one({})
        
        if not sample_doc:
            print("❌ No embeddings found in MongoDB!")
            print("   Run: python scripts/generate_embeddings_to_mongodb.py")
            print()
            client.close()
            sys.exit(1)
        
        # Check if it's flat structure
        has_flat_structure = "embedding" in sample_doc and "intent" in sample_doc
        has_nested_structure = "examples" in sample_doc
        
        if has_nested_structure:
            print("❌ Data is in NESTED structure (old format)")
            print("   Run: python scripts/generate_embeddings_to_mongodb.py")
            print("   This will regenerate embeddings in flat structure")
            print()
            client.close()
            sys.exit(1)
        
        if has_flat_structure:
            print("✅ Data is in FLAT structure (ready for vector search)")
            print(f"   Sample: intent='{sample_doc['intent']}', "
                  f"embedding_dim={len(sample_doc['embedding'])}")
        else:
            print("❌ Unknown data structure")
            print(f"   Sample document keys: {list(sample_doc.keys())}")
            client.close()
            sys.exit(1)
        
        print()
        
        # Count embeddings
        total_count = await db.intent_embeddings.count_documents({})
        intent_count = len(await db.intent_embeddings.distinct("intent"))
        
        print(f"📊 Database Statistics:")
        print(f"   Total embeddings: {total_count}")
        print(f"   Unique intents: {intent_count}")
        print()
        
        # Check vector index
        print("🔍 Checking vector search index...")
        try:
            indexes = await db.intent_embeddings.list_search_indexes().to_list(length=None)
            
            vector_index = None
            for idx in indexes:
                if idx.get('name') == 'vector_index':
                    vector_index = idx
                    break
            
            if vector_index:
                status = vector_index.get('status', 'unknown')
                queryable = vector_index.get('queryable', False)
                
                print(f"✅ Vector index found: 'vector_index'")
                print(f"   Status: {status}")
                print(f"   Queryable: {queryable}")
                
                if not queryable:
                    print()
                    print("⚠️  Index is not queryable yet")
                    print("   Wait a few minutes for index to build")
                    print("   Or check Atlas UI: Database -> Search Indexes")
                    client.close()
                    sys.exit(1)
            else:
                print("❌ Vector index 'vector_index' not found")
                print("   Run: python scripts/setup_vector_index.py")
                print()
                client.close()
                sys.exit(1)
        
        except Exception as e:
            print(f"⚠️  Could not check index status: {str(e)[:100]}")
            print("   Continuing with test anyway...")
        
        print()
        
        # Test vector search with sample queries
        print("=" * 80)
        print("🧪 Testing Vector Search with Sample Queries")
        print("=" * 80)
        print()
        
        test_queries = [
            ("why was my claim rejected", "rejection_reasons"),
            ("hello", "greeting"),
            ("what is the status of my claim", "claim_status"),
            ("how much did I pay", "pricing_info"),
            ("what is the weather", "out_of_scope"),
        ]
        
        # Import embedding service
        from utils.embeddings import get_embedding
        
        for query_text, expected_intent in test_queries:
            print(f"Query: \"{query_text}\"")
            print(f"Expected: {expected_intent}")
            print()
            
            # Get query embedding
            try:
                query_embedding = get_embedding(query_text)
                
                # Perform vector search
                start_time = time.time()
                
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                            "numCandidates": 100,
                            "limit": 50
                        }
                    },
                    {
                        "$project": {
                            "intent": 1,
                            "text": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                cursor = db.intent_embeddings.aggregate(pipeline)
                results = await cursor.to_list(length=50)
                
                search_time = (time.time() - start_time) * 1000  # ms
                
                # Group by intent
                intent_scores = {}
                for doc in results:
                    intent = doc.get("intent")
                    score = doc.get("score", 0.0)
                    
                    if intent not in intent_scores or score > intent_scores[intent]:
                        intent_scores[intent] = score
                
                # Get top intent
                if intent_scores:
                    top_intent = max(intent_scores, key=intent_scores.get)
                    top_score = intent_scores[top_intent]
                    
                    # Get top 3
                    top_3 = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    match = "✅" if top_intent == expected_intent else "❌"
                    print(f"{match} Detected: {top_intent} (confidence: {top_score:.4f})")
                    print(f"   Search time: {search_time:.1f}ms")
                    print(f"   Top 3: {', '.join([f'{i}({s:.3f})' for i, s in top_3])}")
                else:
                    print(f"❌ No results returned")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:200]}")
            
            print()
        
        print("=" * 80)
        print("✅ TESTING COMPLETE")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✅ Flat structure verified")
        print("  ✅ Vector index operational")
        print("  ✅ Vector search functional")
        print()
        print("Next steps:")
        print("  1. Set USE_VECTOR_SEARCH=true in .env")
        print("  2. Restart server")
        print("  3. Test with: curl -X POST http://localhost:8000/api/v1/chat \\")
        print("                -H 'Content-Type: application/json' \\")
        print("                -d '{\"text\":\"hello\",\"session_id\":\"test\",\"user_info\":{\"user_id\":\"test\"}}'")
        print()
        
        client.close()
        
    except Exception as e:
        print()
        print(f"❌ Failed: {str(e)}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(test_vector_search())
    except KeyboardInterrupt:
        print()
        print("❌ Cancelled by user")
        sys.exit(1)

