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
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # LangSmith (optional)
    langsmith_api_key: str | None = None  # Added to avoid ValidationError when env var is present

    # Agent
    confidence_threshold: float = 0.7

    # Safety
    enable_safety_precheck: bool = True
    enable_safety_postcheck: bool = True

    # Cache
    enable_semantic_cache: bool = True

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
