#!/usr/bin/env python3
"""
List MongoDB Databases - Check what databases are accessible

This script connects to MongoDB and lists all databases that the current user can access.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from motor.motor_asyncio import AsyncIOMotorClient
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def list_databases():
    """List all databases accessible to the current user"""
    print("=" * 80)
    print("MongoDB Database Listing")
    print("=" * 80)
    
    # Get MongoDB connection details from config
    connection_string = settings.mongodb_connection_string
    database_name = settings.mongodb_database_name
    
    # Mask connection string for display
    masked_conn = connection_string
    if '@' in connection_string:
        parts = connection_string.split('@')
        if len(parts) == 2:
            user_pass = parts[0]
            if ':' in user_pass:
                user, _ = user_pass.split(':', 1)
                masked_conn = f"{user}:****@{parts[1]}"
    
    print(f"\n📊 Configuration:")
    print(f"   Connection: {masked_conn}")
    print(f"   Target Database: {database_name}")
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
        
        # List all databases
        print("2. Listing accessible databases...")
        try:
            db_list = await client.list_database_names()
            print(f"   ✅ Found {len(db_list)} database(s):")
            print()
            for db_name in sorted(db_list):
                marker = "👉" if db_name == database_name else "  "
                print(f"   {marker} {db_name}")
            print()
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                print("   ⚠️  Permission denied: Cannot list databases")
                print("   (This is normal if your user only has access to specific databases)")
                print()
            else:
                raise
        
        # Check if target database exists and show collections
        print(f"3. Checking target database '{database_name}'...")
        db = client[database_name]
        
        try:
            # Try to list collections (this will work even if we can't list all databases)
            collections = await db.list_collection_names()
            print(f"   ✅ Database '{database_name}' is accessible")
            print(f"   ✅ Found {len(collections)} collection(s):")
            print()
            for col_name in sorted(collections):
                collection = db[col_name]
                count = await collection.count_documents({})
                print(f"      - {col_name}: {count} document(s)")
            print()
        except Exception as e:
            if "not authorized" in str(e).lower():
                print(f"   ❌ Permission denied: Cannot access database '{database_name}'")
                print("   Your user may not have access to this database")
            else:
                print(f"   ❌ Error accessing database: {e}")
            print()
        
        # Try to get database stats
        print("4. Database Statistics:")
        try:
            stats = await db.command("dbStats")
            print(f"   Database: {stats.get('db', database_name)}")
            print(f"   Collections: {stats.get('collections', 0)}")
            print(f"   Data Size: {stats.get('dataSize', 0):,} bytes")
            print(f"   Storage Size: {stats.get('storageSize', 0):,} bytes")
            print()
        except Exception as e:
            print(f"   ⚠️  Could not get stats: {e}")
            print()
        
        print("=" * 80)
        print("✅ Database listing complete!")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ FAILED: Could not list databases")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print()
        
        if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
            print("🔒 PERMISSIONS ISSUE")
            print()
            print("Your MongoDB user may not have permission to list databases.")
            print("This is common when users only have access to specific databases.")
            print()
            print("However, you can still access databases you have permissions for.")
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
        sys.exit(1)
    
    success = asyncio.run(list_databases())
    sys.exit(0 if success else 1)

