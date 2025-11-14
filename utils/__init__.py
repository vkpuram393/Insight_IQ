"""
Utils package for testing individual components and serialization helpers
"""

# Serialization helpers are available at utils.serialization
# Import test router only when needed to avoid fastapi dependency issues
try:
    from utils.test_endpoints import router
    __all__ = ["router"]
except ImportError:
    __all__ = []

