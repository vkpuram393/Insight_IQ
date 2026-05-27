#!/usr/bin/env python3
"""
Setup MongoDB Collections - Drop Old and Create New

This script:
1. Drops the old "Logs" collection (uppercase) if it exists
2. Creates all required collections (lowercase, following MongoDB standards)
3. Inserts one test document in each collection to verify everything works

Collections created:
- logs: Audit logs
- exceptions: Error/exception logs
- events: Telemetry events
- requests: Request/response cycles
- conversation_history: Conversation history

Usage:
    python scripts/setup_mongodb_collections.py
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from motor.motor_asyncio import AsyncIOMotorClient
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def setup_collections():
    """Drop old collections and create new ones with test data"""
    print("=" * 80)
    print("MongoDB Collections Setup")
    print("=" * 80)
    
    # Get MongoDB connection details from config
    connection_string = settings.mongodb_connection_string
    database_name = settings.mongodb_database_name
    
    print(f"\n📊 Configuration:")
    print(f"   Database: {database_name}")
    
    # Mask connection string for display
    masked_conn = connection_string
    if '@' in connection_string:
        parts = connection_string.split('@')
        if len(parts) == 2:
            user_pass = parts[0]
            if ':' in user_pass:
                user, _ = user_pass.split(':', 1)
                masked_conn = f"{user}:****@{parts[1]}"
    print(f"   Connection: {masked_conn}")
    print()
    
    client = None
    try:
        # Connect to MongoDB
        print("1. Connecting to MongoDB...")
        client = AsyncIOMotorClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True
        )
        
        # Test connection
        await client.admin.command('ping')
        print("   ✅ Connected successfully")
        print()
        
        # Get database
        db = client[database_name]
        
        # Step 1: Drop old "Logs" collection if it exists
        print("2. Checking for old 'Logs' collection (uppercase)...")
        collection_names = await db.list_collection_names()
        
        if "Logs" in collection_names:
            print("   ⚠️  Found old 'Logs' collection (uppercase)")
            print("   🗑️  Dropping 'Logs' collection...")
            await db.drop_collection("Logs")
            print("   ✅ Dropped 'Logs' collection")
        else:
            print("   ✅ No old 'Logs' collection found")
        print()
        
        # Step 2: Create all collections and indexes
        collections_config = [
            {
                "name": "logs",
                "description": "Audit logs",
                "indexes": ["session_id", "request_id", "node_name", "timestamp"]
            },
            {
                "name": "exceptions",
                "description": "Error/exception logs",
                "indexes": ["session_id", "request_id", "node_name", "timestamp"]
            },
            {
                "name": "events",
                "description": "Telemetry events",
                "indexes": ["session_id", "event_type", "timestamp"]
            },
            {
                "name": "requests",
                "description": "Request/response cycles",
                "indexes": ["session_id", "user_id", "timestamp"]
            },
            {
                "name": "conversation_history",
                "description": "Conversation history (nested per user_session)",
                "indexes": ["session_id", "user_id", "updated_at"]
                # user_session sparse index created separately below
            },
            {
                "name": "Response_Feedback",
                "description": "User feedback on assistant responses",
                "indexes": ["response_id", "user_id", "created_at"]
            }
        ]
        
        print("3. Creating collections and indexes...")
        print()
        
        for col_config in collections_config:
            col_name = col_config["name"]
            print(f"   📦 {col_name} ({col_config['description']})...")
            
            # Check if collection exists
            if col_name in collection_names:
                print(f"      ⚠️  Collection '{col_name}' already exists")
            else:
                # Create collection explicitly
                await db.create_collection(col_name)
                print(f"      ✅ Created collection '{col_name}'")
            
            # Create indexes
            collection = db[col_name]
            for index_field in col_config["indexes"]:
                try:
                    await collection.create_index(index_field)
                    print(f"      ✅ Index created: {index_field}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"      ⚠️  Index '{index_field}' already exists")
                    else:
                        print(f"      ⚠️  Could not create index '{index_field}': {e}")
            print()
        
        # Sparse index for conversation_history.user_session (must be sparse — field absent when user_session=None)
        print("   Creating sparse index on conversation_history.user_session...")
        try:
            await db.conversation_history.create_index("user_session", sparse=True)
            print("      ✅ Sparse index created: user_session")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("      ⚠️  Sparse index 'user_session' already exists")
            else:
                print(f"      ⚠️  Could not create sparse index 'user_session': {e}")
        print()

        # Step 3: Insert test documents
        print("4. Inserting test documents...")
        print()
        
        test_session_id = f"test-session-{uuid.uuid4().hex[:8]}"
        test_request_id = f"test-request-{uuid.uuid4().hex[:8]}"
        test_user_id = "test-user-123"
        now = datetime.utcnow()
        
        # Test document for logs
        print("   📝 Inserting test document in 'logs'...")
        logs_doc = {
            "_id": str(uuid.uuid4()),
            "session_id": test_session_id,
            "request_id": test_request_id,
            "node_name": "test_setup",
            "event_type": "test_event",
            "data": {"test": True, "message": "Test log entry"},
            "timestamp": now,
            "user_id": test_user_id,
            "created_at": now
        }
        await db.logs.insert_one(logs_doc)
        print("      ✅ Test document inserted in 'logs'")
        print()
        
        # Test document for exceptions
        print("   📝 Inserting test document in 'exceptions'...")
        exceptions_doc = {
            "_id": str(uuid.uuid4()),
            "error_code": "TEST_ERROR",
            "category": "test",
            "severity": "low",
            "message": "Test exception message",
            "user_message": "This is a test exception",
            "session_id": test_session_id,
            "request_id": test_request_id,
            "node_name": "test_setup",
            "stacktrace": "Test stacktrace",
            "metadata": {"test": True},
            "timestamp": now,
            "user_id": test_user_id,
            "created_at": now
        }
        await db.exceptions.insert_one(exceptions_doc)
        print("      ✅ Test document inserted in 'exceptions'")
        print()
        
        # Test document for events
        print("   📝 Inserting test document in 'events'...")
        events_doc = {
            "_id": str(uuid.uuid4()),
            "event_type": "test_event",
            "session_id": test_session_id,
            "data": {"test": True, "message": "Test event"},
            "timestamp": now,
            "user_id": test_user_id,
            "created_at": now
        }
        await db.events.insert_one(events_doc)
        print("      ✅ Test document inserted in 'events'")
        print()
        
        # Test document for requests
        print("   📝 Inserting test document in 'requests'...")
        requests_doc = {
            "_id": str(uuid.uuid4()),
            "session_id": test_session_id,
            "user_id": test_user_id,
            "user_text": "Test request",
            "intent": "test_intent",
            "confidence": 0.95,
            "response": "Test response",
            "metadata": {"test": True, "duration_ms": 100},
            "timestamp": now,
            "created_at": now
        }
        await db.requests.insert_one(requests_doc)
        print("      ✅ Test document inserted in 'requests'")
        print()
        
        # Test document for conversation_history — new nested schema keyed by user_session
        print("   📝 Inserting test document in 'conversation_history'...")
        conversation_doc = {
            "_id": "test-user-session-001",
            "user_session": "test-user-session-001",
            "session_id": "session-test-setup-check",
            "user_id": "setup_test_user",
            "created_at": now,
            "updated_at": now,
            "conversation_history": [
                {
                    "role": "user",
                    "content": "Test message from setup script",
                    "timestamp": now.isoformat()
                },
                {
                    "role": "assistant",
                    "content": "Test response from setup script",
                    "timestamp": now.isoformat(),
                    "response_id": None
                }
            ]
        }
        await db.conversation_history.insert_one(conversation_doc)
        print("      ✅ Test document inserted in 'conversation_history'")
        print()

        # Test document for Response_Feedback
        print("   📝 Inserting test document in 'Response_Feedback'...")
        response_feedback_doc = {
            "_id": str(uuid.uuid4()),
            "response_id": str(uuid.uuid4()),
            "user_id": test_user_id,
            "session_id": test_session_id,
            "feedback_type": "THUMBSUP",
            "created_at": now
        }
        await db.Response_Feedback.insert_one(response_feedback_doc)
        print("      ✅ Test document inserted in 'Response_Feedback'")
        print()
        
        # Step 4: Verify all collections have data
        print("5. Verifying collections...")
        print()
        
        verification_results = []
        for col_config in collections_config:
            col_name = col_config["name"]
            collection = db[col_name]
            count = await collection.count_documents({})
            verification_results.append((col_name, count))
            print(f"   ✅ '{col_name}': {count} document(s)")
        
        print()
        print("=" * 80)
        print("✅ SUCCESS: All collections setup and tested!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"   Database: {database_name}")
        print(f"   Collections created: {len(collections_config)}")
        print(f"   Test session ID: {test_session_id}")
        print()
        print("Collections:")
        for col_name, count in verification_results:
            print(f"   - {col_name}: {count} document(s)")
        print()
        print("✅ All collections are ready to use!")
        print()
        print("Note: Test documents were inserted with session_id:")
        print(f"   {test_session_id}")
        print("   conversation_history keyed by _id='test-user-session-001' (new nested schema)")
        print("   You can query/delete these test documents if needed.")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ FAILED: Could not setup collections")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print()
        
        if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
            print("🔒 PERMISSIONS ISSUE")
            print()
            print("Your MongoDB user does not have required permissions.")
            print()
            print("Required Permissions:")
            print(f"  - 'readWrite' role on database: {database_name}")
            print("  - Permission to drop collections")
            print("  - Permission to create collections and indexes")
        elif "authentication failed" in str(e).lower():
            print("🔐 AUTHENTICATION ISSUE")
            print()
            print("MongoDB authentication failed. Please check:")
            print("  1. Username and password in connection string")
            print("  2. User has access to the database")
        elif "timeout" in str(e).lower() or "network" in str(e).lower():
            print("🌐 CONNECTION ISSUE")
            print()
            print("Could not connect to MongoDB server. Please check:")
            print("  1. Connection string is correct")
            print("  2. Network connectivity to MongoDB server")
            print("  3. Firewall rules allow connection")
        else:
            print("Troubleshooting:")
            print("  1. Verify MongoDB connection string in .env file")
            print("  2. Check that MongoDB server is running")
            print("  3. Verify database name is correct")
        
        print()
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if client:
            client.close()
            print("   Connection closed")


if __name__ == "__main__":
    # Check if MongoDB is configured
    if settings.persistence_store_type != "mongodb":
        print("⚠️  WARNING: PERSISTENCE_STORE_TYPE is not set to 'mongodb'")
        print(f"   Current value: {settings.persistence_store_type}")
        print()
        print("To use this script:")
        print("  1. Set PERSISTENCE_STORE_TYPE=mongodb in .env file")
        print("  2. Set MONGODB_CONNECTION_STRING in .env file")
        print("  3. Set MONGODB_DATABASE_NAME in .env file")
        print()
        sys.exit(1)
    
    success = asyncio.run(setup_collections())
    sys.exit(0 if success else 1)

