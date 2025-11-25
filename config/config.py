"""Configuration - All settings"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # LLM
    openai_api_key: str = "mock"
    use_mock_llm: bool = False  # Set to False to use real Google Gemini LLM
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

    # Azure OpenAI Embeddings (for embedding-based classifier)
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""  # Optional if using Azure AD
    azure_openai_chat_model: str = "gpt-4"
    azure_openai_embedding_model: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"
    
    # Azure AD Service Principal (alternative to API key)
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""

    # Agent
    confidence_threshold: float = 0.6  # Low confidence queries (< 0.6) route to response_agent (LLM)
    conversation_history_limit: int = 5  # Number of past conversations to include in response generation
    use_cvs_intent_classifier: bool = True  # Set to True to use Ahmed's production-ready CVS classifier (28+ intents)
    use_embedding_classifier: bool = True  # Set to True to use embedding-based classifier instead of keyword-based (requires use_cvs_intent_classifier=True)

    # Safety
    enable_safety_precheck: bool = True  # Match remote MVP-1
    enable_safety_postcheck: bool = True  # Match remote MVP-1
    
    # LLM Judge (TEMPORARY - for testing only, disables LLM judge path when True)
    temporarily_disable_llm_judge_path_for_testing: bool = False  # Set to False when LLM judge implementation is complete

    # Cache
    enable_semantic_cache: bool = True
    
    # Orchestrator
    remove_punctuation_in_normalization: bool = True

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

    # API Configuration
    swagger_url: str ="https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com" # Base URL for external APIs

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
