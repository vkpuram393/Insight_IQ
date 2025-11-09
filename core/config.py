"""Configuration - All settings"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # LLM
    openai_api_key: str = "mock"
    use_mock_llm: bool = True
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 2048
    project_id: str = "pbm-poc-coderev-genai-poc"
    location: str = "us-central1"

    # LangSmith (optional)
    langsmith_api_key: str | None = None  # Added to avoid ValidationError when env var is present

    # Langfuse (optional)
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_host: str | None = None

    # SSL Certificates (for CVS proxy/Zscaler)
    ssl_cert_file: str | None = None
    requests_ca_bundle: str | None = None

    # Agent
    confidence_threshold: float = 0.7

    # Safety
    enable_safety_precheck: bool = True
    enable_safety_postcheck: bool = True

    # Cache
    enable_semantic_cache: bool = True

    # Memory Store (cache and session memory)
    memory_store_type: str = "inmemory"  # Options: "inmemory", "redis", "memorystore"
    memory_store_host: str | None = None
    memory_store_port: int = 6379

    # Persistence Store (telemetry and analytics)
    persistence_store_type: str = "sqlite"  # Options: "sqlite", "firestore", "bigquery"
    telemetry_db_path: str = "data/telemetry.db"
    enable_telemetry: bool = True

    # Checkpoint
    checkpoint_db_path: str = "checkpoints.db"
    enable_checkpointing: bool = False  # disable to avoid async saver issues during local debugging

    # App
    environment: str = "development"
    debug: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
