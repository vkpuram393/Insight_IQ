"""
Pytest configuration and fixtures for test database isolation

This ensures tests use a separate database from the running server,
preventing SQLite lock conflicts.
"""

import pytest
import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import settings
from persistence import PersistenceStoreFactory


# Test database path (separate from production)
TEST_DB_PATH = "data/telemetry_test.db"


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Set up test database before all tests run.
    
    This fixture:
    1. Overrides the database path to use a test database
    2. Cleans up the test database after all tests complete
    """
    # Store original path
    original_path = settings.telemetry_db_path
    
    # Override with test database path
    settings.telemetry_db_path = TEST_DB_PATH
    
    # Ensure test data directory exists
    Path(TEST_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Reset persistence store instance to force reinitialization with test DB
    PersistenceStoreFactory._instance = None
    
    yield
    
    # Cleanup: Close any open connections first
    try:
        # Close persistence store instance if it exists
        if PersistenceStoreFactory._instance is not None:
            # Run async close in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(PersistenceStoreFactory.close_instance())
            finally:
                loop.close()
    except Exception:
        pass  # Ignore cleanup errors
    
    # Reset instance
    PersistenceStoreFactory._instance = None
    
    # Restore original path
    settings.telemetry_db_path = original_path
    
    # Clean up test database file after all tests (with retry for file locks)
    if os.path.exists(TEST_DB_PATH):
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Also try to remove WAL and SHM files
                wal_path = TEST_DB_PATH + "-wal"
                shm_path = TEST_DB_PATH + "-shm"
                if os.path.exists(wal_path):
                    os.remove(wal_path)
                if os.path.exists(shm_path):
                    os.remove(shm_path)
                os.remove(TEST_DB_PATH)
                break
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    pass  # Give up after max retries
