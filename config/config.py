"""Configuration - All settings"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # LLM
    openai_api_key: str = "mock"  # ⚠️ Overridden by OPENAI_API_KEY env var
    use_mock_llm: bool = False  # Set to False to use real Google Gemini LLM
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 2048
    project_id: str = "pbm-nonprod-myclaims"  # ⚠️ Overridden by PROJECT_ID env var
    location: str = "us-central1"  # ⚠️ Overridden by LOCATION env var

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
    azure_openai_endpoint: str = ""  # ⚠️ Overridden by AZURE_OPENAI_ENDPOINT env var
    azure_openai_key: str = ""  # ⚠️ Overridden by AZURE_OPENAI_KEY env var (Optional if using Azure AD)
    azure_openai_chat_model: str = "gpt-4"
    azure_openai_embedding_model: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"
    
    # Azure AD Service Principal (alternative to API key)
    azure_tenant_id: str = ""  # ⚠️ Overridden by AZURE_TENANT_ID env var
    azure_client_id: str = ""  # ⚠️ Overridden by AZURE_CLIENT_ID env var
    azure_client_secret: str = ""  # ⚠️ Overridden by AZURE_CLIENT_SECRET env var
    
    # Embeddings Provider Selection (uses same project_id/location as Gemini LLM)
    use_google_embeddings: bool = False  # Set to True to use Google Cloud text-embedding-005 instead of Azure

    # Agent
    confidence_threshold: float = 0.6  # Not used. It is bypassed in domain_config.json. Low confidence queries (< 0.6) route to response_agent (LLM)
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
    # ⚠️ IMPORTANT: This default is OVERRIDDEN by PERSISTENCE_STORE_TYPE in .env file
    # The .env file takes precedence - set PERSISTENCE_STORE_TYPE=sqlite or PERSISTENCE_STORE_TYPE=mongodb in .env
    # Options:
    #   - "sqlite": Local SQLite database (no server required, good for development)
    #   - "mongodb": MongoDB database (requires MongoDB server, set connection details below)
    #   - "firestore": Google Cloud Firestore (future)
    #   - "bigquery": Google BigQuery (future)
    persistence_store_type: str = "sqlite"  # Default fallback (overridden by .env)
    telemetry_db_path: str = "data/telemetry.db"  # Only used for SQLite
    
    # MongoDB Configuration (only used when persistence_store_type="mongodb")
    # Connection string format: mongodb+srv://<username>:<password>@<cluster>/<database>?retryWrites=true&w=majority
    # Or for direct IP: mongodb://<username>:<password>@<ip1>:<port>,<ip2>:<port>/<database>?replicaSet=<replica_set>
    mongodb_connection_string: str = "mongodb://localhost:27017"  # Default local MongoDB
    mongodb_database_name: str = "myclaims-DEV"  # Database name (myclaims-DEV, myclaims-QA, myClaims-UAT, myClaims-PT)
    
    enable_telemetry: bool = True

    # Checkpoint
    checkpoint_db_path: str = "checkpoints.db"
    enable_checkpointing: bool = True  # Enable to persist state across turns for follow-up questions

    # App
    environment: str = "development"  # ⚠️ Overridden by ENVIRONMENT env var
    debug: bool = True  # ⚠️ Overridden by DEBUG env var

    # API Configuration
    swagger_url: str = "https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com"  # ⚠️ Overridden by SWAGGER_URL env var - Base URL for external APIs

    # Streaming Configuration
    enable_streaming: bool = True  # Master switch for streaming feature
    streaming_chunk_size: int = 50  # Characters per chunk when streaming response
    streaming_delay_ms: int = 0  # Artificial delay between chunks (0 for production, 30-50 for demo)
    stream_node_updates: bool = True  # Stream node status updates for observability
    
    # Control which nodes send user-facing status updates (reduces noise, improves UX)
    # Other nodes are still logged/tracked for telemetry, just not shown to end users
    # Industry best practice: Show only significant milestones (5-6 key steps)
    stream_user_facing_nodes: list = [
        "orchestrator",          # Initial processing
        "safety_precheck",       # Privacy/safety check (builds trust)
        "intent_agent",          # AI understanding the question
        "call_claims_tool",      # Data retrieval (most important to users)
        "response_agent"         # Final response generation
    ]

    class Config:
        """
        Configuration loading priority (Pydantic Settings):
        1. Environment variables (HIGHEST - used in production deployments)
        2. .env file (for local development only)
        3. Default values in this class (fallback)
        
        PRODUCTION DEPLOYMENT:
        - Do NOT use .env files in production
        - Set environment variables directly in your deployment platform:
          * Kubernetes: ConfigMaps/Secrets
          * Docker: -e flags or docker-compose.yml
          * Cloud Run/GKE: Environment variables in deployment config
          * GitHub Actions: Secrets and env: section
        
        LOCAL DEVELOPMENT:
        - Use .env file for convenience
        - Can create .env.dev, .env.qa, .env.prod for different environments
        - Switch by: cp .env.qa .env (or use ENVIRONMENT variable)
        """
        env_file = ".env"  # Only used if file exists (ignored in production)
        case_sensitive = False

settings = Settings()
