#!/usr/bin/env python3
"""Quick diagnostic script to test SQLite logging"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import settings
from persistence import PersistenceStoreFactory

import pytest

@pytest.mark.asyncio
async def test():
    print("=" * 60)
    print("DIAGNOSTIC: SQLite Logging")
    print("=" * 60)
    print(f"Telemetry enabled: {settings.enable_telemetry}")
    print(f"Persistence store type: {settings.persistence_store_type}")
    print(f"DB path: {settings.telemetry_db_path}")
    
    try:
        store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        print(f"Store instance: {type(store).__name__}")
        print(f"Store DB path: {store.db_path}")
        
        # Try to log something
        log_id = await store.log_audit(
            session_id="test-session-123",
            request_id="test-request-456",
            user_id="test-user-789",
            node_name="test_node",
            event_type="test_event",
            data={"test": "data", "message": "This is a test log entry"}
        )
        print(f"✅ Log created! Log ID: {log_id}")
        
        # Check if it's in the database
        import sqlite3
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables: {tables}")
        
        # Check logs count
        if 'logs' in tables:
            cursor.execute("SELECT COUNT(*) FROM logs")
            count = cursor.fetchone()[0]
            print(f"✅ Total logs: {count}")
            
            if count > 0:
                cursor.execute("SELECT log_id, node_name, event_type, timestamp FROM logs ORDER BY timestamp DESC LIMIT 5")
                print(f"\nRecent logs:")
                for row in cursor.fetchall():
                    print(f"  - {row[1]} ({row[2]}) at {row[3]}")
        else:
            print("❌ 'logs' table not found!")
        
        conn.close()
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())


