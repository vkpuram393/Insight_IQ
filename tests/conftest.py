"""
Pytest configuration and fixtures for test database isolation

This ensures tests use a separate database from the running server,
preventing SQLite lock conflicts.
"""

import pytest
import pytest_asyncio
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


# Test database paths (separate from production)
TEST_DB_PATH = "data/telemetry_test.db"
TEST_CHECKPOINT_DB_PATH = "data/checkpoints_test.db"


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Set up test database before all tests run.
    
    This fixture:
    1. Overrides the database path to use a test database
    2. Overrides the checkpoint database path
    3. Cleans up the test databases after all tests complete
    """
    # Store original paths
    original_path = settings.telemetry_db_path
    original_checkpoint_path = settings.checkpoint_db_path
    
    # Override with test database paths
    settings.telemetry_db_path = TEST_DB_PATH
    settings.checkpoint_db_path = TEST_CHECKPOINT_DB_PATH
    
    # Ensure test data directory exists
    Path(TEST_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Reset persistence store instance to force reinitialization with test DB
    PersistenceStoreFactory._instance = None
    
    yield
    
    # Cleanup: Close any open connections first
    # Note: We skip async cleanup here to avoid event loop conflicts with pytest-asyncio.
    # The persistence store connections will be cleaned up when the process exits.
    # If you need explicit cleanup, consider making it synchronous or using a different approach.
    pass
    
    # Reset instance
    PersistenceStoreFactory._instance = None
    
    # Restore original paths
    settings.telemetry_db_path = original_path
    settings.checkpoint_db_path = original_checkpoint_path
    
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
    
    # Clean up test checkpoint database file after all tests (with retry for file locks)
    if os.path.exists(TEST_CHECKPOINT_DB_PATH):
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Also try to remove WAL and SHM files
                wal_path = TEST_CHECKPOINT_DB_PATH + "-wal"
                shm_path = TEST_CHECKPOINT_DB_PATH + "-shm"
                if os.path.exists(wal_path):
                    os.remove(wal_path)
                if os.path.exists(shm_path):
                    os.remove(shm_path)
                os.remove(TEST_CHECKPOINT_DB_PATH)
                break
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    pass  # Give up after max retries


@pytest_asyncio.fixture(scope="function")
async def init_graph_for_test():
    """
    Initialize graph for streaming tests and clean up after each test.
    
    This fixture ensures the graph is properly initialized with the async
    SQLite checkpointer for streaming tests.
    """
    from langgraph_agent import init_graph, close_graph
    
    # Initialize graph (this enters the async context manager)
    await init_graph()
    
    yield
    
    # Close graph after test to reset connection
    await close_graph()
