#!/usr/bin/env python3
"""
Create 'logs' Collection in MongoDB

This script creates a collection named "logs" in the MongoDB instance
configured in the .env file.

Usage:
    python scripts/create_logs_collection.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from motor.motor_asyncio import AsyncIOMotorClient
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def create_logs_collection():
    """Create 'logs' collection in MongoDB"""
    print("=" * 80)
    print("Create 'logs' Collection in MongoDB")
    print("=" * 80)
    
    # Get MongoDB connection details from config
    connection_string = settings.mongodb_connection_string
    database_name = settings.mongodb_database_name
    
    print(f"\n📊 Configuration:")
    print(f"   Database: {database_name}")
    print(f"   Collection: logs")
    
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
        
        # Check if collection already exists
        print("2. Checking if 'logs' collection exists...")
        collection_names = await db.list_collection_names()
        
        if "logs" in collection_names:
            print("   ⚠️  Collection 'logs' already exists")
            print()
            print("   Collection details:")
            collection = db.logs
            count = await collection.count_documents({})
            print(f"   - Document count: {count}")
            print()
            print("   ✅ Collection is ready to use")
            return True
        
        # Create collection
        print("   Collection 'logs' does not exist")
        print()
        print("3. Creating 'logs' collection...")
        
        # Create collection explicitly
        await db.create_collection("logs")
        print("   ✅ Collection 'logs' created successfully")
        print()
        
        # Create indexes for better performance
        print("4. Creating indexes...")
        collection = db.logs
        
        try:
            await collection.create_index("session_id")
            print("   ✅ Index created: session_id")
        except Exception as e:
            print(f"   ⚠️  Could not create session_id index: {e}")
        
        try:
            await collection.create_index("request_id")
            print("   ✅ Index created: request_id")
        except Exception as e:
            print(f"   ⚠️  Could not create request_id index: {e}")
        
        try:
            await collection.create_index("node_name")
            print("   ✅ Index created: node_name")
        except Exception as e:
            print(f"   ⚠️  Could not create node_name index: {e}")
        
        try:
            await collection.create_index("timestamp")
            print("   ✅ Index created: timestamp")
        except Exception as e:
            print(f"   ⚠️  Could not create timestamp index: {e}")
        
        print()
        print("=" * 80)
        print("✅ SUCCESS: 'logs' collection created successfully!")
        print("=" * 80)
        print()
        print("Collection Details:")
        print(f"   Database: {database_name}")
        print(f"   Collection: logs")
        print(f"   Indexes: session_id, request_id, node_name, timestamp")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ FAILED: Could not create 'logs' collection")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print()
        
        if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
            print("🔒 PERMISSIONS ISSUE")
            print()
            print("Your MongoDB user does not have permission to create collections.")
            print()
            print("Required Permissions:")
            print(f"  - 'readWrite' role on database: {database_name}")
            print()
            print("Note: Collections are also created automatically when you insert")
            print("the first document, so you may still be able to use the collection.")
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
    
    success = asyncio.run(create_logs_collection())
    sys.exit(0 if success else 1)

