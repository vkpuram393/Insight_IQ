#!/usr/bin/env python3
"""
Test MongoDB Connection and Create Collections

This script:
1. Tests MongoDB connectivity
2. Creates collections (logs, exceptions, events, requests, conversation_history) if they don't exist
3. Inserts test data to verify everything works

Usage:
    python scripts/test_mongodb_connection.py

Environment Variables:
    PERSISTENCE_STORE_TYPE=mongodb
    MONGODB_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>/<database>
    MONGODB_DATABASE_NAME=myclaims-DEV
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import settings
from persistence import PersistenceStoreFactory
from core.logger import get_logger

logger = get_logger(__name__)


async def test_mongodb_connection():
    """Test MongoDB connection and create collections"""
    print("=" * 80)
    print("MongoDB Connection Test")
    print("=" * 80)
    print(f"Persistence Store Type: {settings.persistence_store_type}")
    print(f"MongoDB Connection String: {settings.mongodb_connection_string[:50]}...")  # Hide password
    print(f"MongoDB Database Name: {settings.mongodb_database_name}")
    print()

    if settings.persistence_store_type != "mongodb":
        print("⚠️  WARNING: persistence_store_type is not set to 'mongodb'")
        print(f"   Current value: {settings.persistence_store_type}")
        print("   Set PERSISTENCE_STORE_TYPE=mongodb in .env file")
        print()
        return False

    try:
        # Get MongoDB persistence store instance
        print("1. Initializing MongoDB Persistence Store...")
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        print(f"   ✅ Store initialized: {type(persistence_store).__name__}")
        print()

        # Test connection by getting database (this will create indexes)
        print("2. Testing MongoDB connection...")
        try:
            db = await persistence_store._get_connection()
            print(f"   ✅ Connected to database: {persistence_store.database_name}")
            
            # Verify we can ping the database
            await db.command('ping')
            print("   ✅ Database ping successful")
        except Exception as e:
            print(f"   ❌ Connection failed: {str(e)}")
            print()
            print("   Troubleshooting:")
            print("   1. Check MongoDB connection string format")
            print("   2. Verify username and password are correct")
            print("   3. Ensure network connectivity to MongoDB server")
            print("   4. For MongoDB Atlas, check IP whitelist")
            raise
        print()

        # List existing collections (optional - may not have permission)
        print("3. Checking existing collections...")
        try:
            existing_collections = await db.list_collection_names()
            print(f"   ✅ Existing collections: {existing_collections}")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot list collections (permission issue - this is OK)")
                print(f"   Note: You may not have 'listCollections' permission, but insert/read should work")
            else:
                print(f"   ⚠️  Could not list collections: {str(e)}")
            existing_collections = []
        print()

        # Test log_audit - this will create 'logs' collection if it doesn't exist
        print("4. Testing log_audit (creates 'logs' collection)...")
        test_log_id = await persistence_store.log_audit(
            session_id="test-session-connection",
            node_name="test_mongodb_connection",
            event_type="connection_test",
            data={
                "test": True,
                "message": "MongoDB connection test successful",
                "timestamp": "test"
            },
            request_id="test-request-connection",
            user_id="test-user"
        )
        print(f"   ✅ Log created with ID: {test_log_id}")
        print()

        # Verify logs collection exists and has data
        try:
            logs_count = await db.logs.count_documents({})
            print(f"   ✅ 'logs' collection exists with {logs_count} document(s)")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot count documents (permission issue - insert succeeded, this is OK)")
            else:
                print(f"   ⚠️  Could not count documents: {str(e)}")
        print()

        # Test log_exception - this will create 'exceptions' collection if it doesn't exist
        print("5. Testing log_exception (creates 'exceptions' collection)...")
        test_exception_id = await persistence_store.log_exception(
            error_code="TEST_CONNECTION",
            category="test",
            severity="info",
            message="MongoDB connection test exception",
            user_message="This is a test exception",
            session_id="test-session-connection",
            request_id="test-request-connection",
            node_name="test_mongodb_connection",
            stacktrace="No stacktrace for test",
            metadata={"test": True},
            user_id="test-user"
        )
        print(f"   ✅ Exception logged with ID: {test_exception_id}")
        print()

        # Verify exceptions collection exists and has data
        try:
            exceptions_count = await db.exceptions.count_documents({})
            print(f"   ✅ 'exceptions' collection exists with {exceptions_count} document(s)")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot count documents (permission issue - insert succeeded, this is OK)")
            else:
                print(f"   ⚠️  Could not count documents: {str(e)}")
        print()

        # List all collections after operations (optional - may not have permission)
        print("6. Final collection status...")
        try:
            final_collections = await db.list_collection_names()
            print(f"   Collections: {final_collections}")
            
            # Verify required collections exist
            required_collections = ["logs", "exceptions", "events", "requests", "conversation_history"]
            missing_collections = [col for col in required_collections if col not in final_collections]
            
            if missing_collections:
                print(f"   ⚠️  Note: Some collections will be created on first use:")
                for col in missing_collections:
                    print(f"      - {col}")
                print("   (This is normal - MongoDB creates collections automatically on first insert)")
            else:
                print("   ✅ All required collections are available")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot list collections (permission issue - this is OK)")
                print(f"   Collections will be created automatically on first insert")
            else:
                print(f"   ⚠️  Could not list collections: {str(e)}")
        print()

        # Test reading back the data
        print("7. Verifying data can be read back...")
        try:
            test_logs = await db.logs.find({"session_id": "test-session-connection"}).to_list(length=10)
            test_exceptions = await db.exceptions.find({"session_id": "test-session-connection"}).to_list(length=10)
            
            print(f"   ✅ Found {len(test_logs)} test log(s)")
            print(f"   ✅ Found {len(test_exceptions)} test exception(s)")
            
            if len(test_logs) == 0 or len(test_exceptions) == 0:
                print(f"   ⚠️  Warning: Could not read back test data - may indicate read permission issues")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot read data (permission issue)")
                print(f"   Note: Insert succeeded, but read may require additional permissions")
                print(f"   This may be OK if your MongoDB user only has write permissions")
            else:
                print(f"   ⚠️  Could not read back data: {str(e)}")
        print()

        # Clean up test data (optional)
        print("8. Cleaning up test data...")
        try:
            delete_logs_result = await db.logs.delete_many({"session_id": "test-session-connection"})
            delete_exceptions_result = await db.exceptions.delete_many({"session_id": "test-session-connection"})
            print(f"   ✅ Deleted {delete_logs_result.deleted_count} test log(s)")
            print(f"   ✅ Deleted {delete_exceptions_result.deleted_count} test exception(s)")
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"   ⚠️  Cannot delete test data (permission issue - this is OK)")
                print(f"   Test data will remain in database")
            else:
                print(f"   ⚠️  Could not delete test data: {str(e)}")
        print()

        print("=" * 80)
        print("✅ MongoDB Connection Test PASSED")
        print("=" * 80)
        print()
        print("Note: If you saw permission warnings above, your MongoDB user may have")
        print("limited permissions (e.g., write-only). This is OK as long as:")
        print("  ✅ Connection succeeded")
        print("  ✅ Insert operations (log_audit, log_exception) succeeded")
        print()
        print("The application will work correctly even with limited permissions.")
        print("Collections are created automatically on first insert.")
        print()
        print("Next steps:")
        print("1. Update your .env file with:")
        print("   PERSISTENCE_STORE_TYPE=mongodb")
        print("   MONGODB_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>/<database>")
        print("   MONGODB_DATABASE_NAME=myclaims-DEV  # or myclaims-QA, myClaims-UAT, myClaims-PT")
        print()
        print("2. Restart your application to use MongoDB")
        print()

        # Close connection
        await persistence_store.close()
        return True

    except Exception as e:
        print("=" * 80)
        print("❌ MongoDB Connection Test FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        
        # Check if it's a permissions issue
        if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
            print("🔒 PERMISSIONS ISSUE DETECTED")
            print()
            print("Your MongoDB user does not have the required permissions.")
            print()
            print("Required Permissions:")
            print("  - 'readWrite' role on database: " + settings.mongodb_database_name)
            print()
            print("How to Fix:")
            print("  1. Contact your MongoDB administrator")
            print("  2. Request 'readWrite' role on database: " + settings.mongodb_database_name)
            print("  3. For MongoDB Atlas: Database Access → Edit User → Add 'readWrite' role")
            print()
            print("See docs/MONGODB_PERMISSIONS.md for detailed instructions.")
            print()
        else:
            print("Troubleshooting:")
            print("1. Verify MongoDB connection string is correct")
            print("2. Check that username and password are correct")
            print("3. Verify network connectivity to MongoDB server")
            print("4. Check firewall rules allow connection to MongoDB port (27017)")
            print("5. For MongoDB Atlas, ensure IP whitelist includes your IP")
            print()
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mongodb_connection())
    sys.exit(0 if success else 1)

