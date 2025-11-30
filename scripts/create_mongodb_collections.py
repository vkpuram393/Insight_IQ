#!/usr/bin/env python3
"""
Create MongoDB Collections Explicitly

This script creates the required MongoDB collections explicitly.
Note: Collections are automatically created on first insert, but this script
can be useful for:
- Pre-creating collections with specific options
- Verifying permissions
- Setting up initial structure

Usage:
    python scripts/create_mongodb_collections.py
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


async def create_collections():
    """Create MongoDB collections explicitly"""
    print("=" * 80)
    print("MongoDB Collection Creation")
    print("=" * 80)
    print(f"Persistence Store Type: {settings.persistence_store_type}")
    print(f"MongoDB Database: {settings.mongodb_database_name}")
    print()

    if settings.persistence_store_type != "mongodb":
        print("⚠️  WARNING: persistence_store_type is not set to 'mongodb'")
        print(f"   Current value: {settings.persistence_store_type}")
        print("   Set PERSISTENCE_STORE_TYPE=mongodb in .env file")
        return False

    try:
        # Get MongoDB persistence store instance
        print("1. Connecting to MongoDB...")
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        db = await persistence_store._get_connection()
        print(f"   ✅ Connected to database: {persistence_store.database_name}")
        print()

        # Collections to create
        collections = [
            "logs",              # Audit logs
            "exceptions",        # Error/exception logs
            "events",            # Telemetry events
            "requests",          # Request/response cycles
            "conversation_history"  # Conversation history
        ]

        print("2. Creating collections...")
        print()

        created_collections = []
        existing_collections = []
        failed_collections = []

        for collection_name in collections:
            try:
                # Check if collection already exists
                collection_list = await db.list_collection_names()
                if collection_name in collection_list:
                    print(f"   ⚠️  Collection '{collection_name}' already exists")
                    existing_collections.append(collection_name)
                    continue

                # Create collection explicitly
                # MongoDB creates collections automatically, but we can create them with options
                await db.create_collection(collection_name)
                print(f"   ✅ Created collection: '{collection_name}'")
                created_collections.append(collection_name)

            except Exception as e:
                if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                    print(f"   ❌ Permission denied for '{collection_name}': {str(e)}")
                    failed_collections.append(collection_name)
                elif "already exists" in str(e).lower():
                    print(f"   ⚠️  Collection '{collection_name}' already exists")
                    existing_collections.append(collection_name)
                else:
                    print(f"   ❌ Failed to create '{collection_name}': {str(e)}")
                    failed_collections.append(collection_name)

        print()

        # Summary
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"✅ Created: {len(created_collections)} collection(s)")
        if created_collections:
            for col in created_collections:
                print(f"   - {col}")

        print(f"⚠️  Already exists: {len(existing_collections)} collection(s)")
        if existing_collections:
            for col in existing_collections:
                print(f"   - {col}")

        if failed_collections:
            print(f"❌ Failed: {len(failed_collections)} collection(s)")
            for col in failed_collections:
                print(f"   - {col}")
            print()
            print("Note: Collections will be created automatically when you insert the first document.")
            print("If you see permission errors, you may need 'readWrite' role on the database.")
            return False

        print()
        print("✅ Collection creation completed successfully!")
        print()
        print("Note: Collections are also created automatically when you insert the first document.")
        print("This explicit creation is optional but can be useful for verification.")

        # Close connection
        await persistence_store.close()
        return True

    except Exception as e:
        print("=" * 80)
        print("❌ Collection Creation Failed")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        
        if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
            print("🔒 PERMISSIONS ISSUE DETECTED")
            print()
            print("Your MongoDB user does not have permission to create collections.")
            print()
            print("Required Permissions:")
            print(f"  - 'readWrite' role on database: {settings.mongodb_database_name}")
            print()
            print("However, collections are created automatically when you insert the first document.")
            print("So you can still use the application - collections will be created on first use.")
            print()
            print("See docs/MONGODB_PERMISSIONS.md for details.")
        else:
            print("Troubleshooting:")
            print("1. Verify MongoDB connection string is correct")
            print("2. Check that username and password are correct")
            print("3. Verify network connectivity to MongoDB server")
            print()
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(create_collections())
    sys.exit(0 if success else 1)

