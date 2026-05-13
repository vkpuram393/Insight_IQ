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
    # Gemini 2.5 Flash max=65,536. Set 32,768 (50%) to accommodate internal thinking + response tokens.
    max_output_tokens: int = 32768
    project_id: str = "pbm-nonprod-myclaims"  # ⚠️ Overridden by PROJECT_ID env var (using old project - you don't have access to pbm-nonprod-myclaims yet)
    location: str = "us-central1"  # ⚠️ Overridden by LOCATION env var
    
    # LLM Thinking Mode - Gemini thinks internally, these settings make it VISIBLE.
    # ⚠️ Keep OFF in production (increases latency/cost). Enable only for debugging.
    enable_thinking_mode: bool = True  # ⚠️ Overridden by ENABLE_THINKING_MODE env var
    log_thoughts_to_mongo: bool = True # Store thoughts in MongoDB. ⚠️ Overridden by LOG_THOUGHTS_TO_MONGO env var

    # LangSmith (optional)
    langsmith_api_key: str | None = None  # Added to avoid ValidationError when env var is present

    # Langfuse (optional)
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_host: str | None = "https://cloud.langfuse.com"

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
    use_google_embeddings: bool = True  # Set to True to use Google Cloud text-embedding-005 instead of Azure

    # Agent
    confidence_threshold: float = 0.6  # Not used. It is bypassed in domain_config.json. Low confidence queries (< 0.6) route to response_agent (LLM)
    conversation_history_limit: int = 100  # Number of past messages (50 turns) to include in response generation and entity extraction
    use_embedding_classifier: bool = True  # True = Embedding-based classifier (semantic), False = Keyword-based classifier (fast)

    # Safety
    enable_safety_precheck: bool = True  # Match remote MVP-1
    enable_safety_postcheck: bool = True  # Match remote MVP-1
    
    # Leakage Detection Mode (for response_safety_pii_postcheck)
    # Options:
    #   - "block": Block response if unexpected PII detected (strict, may cause false positives)
    #   - "log": Log detected entities but don't block (recommended for production)
    #   - "disabled": Skip leakage detection entirely (fastest, masking still protects PHI)
    # Note: HIPAA compliance is achieved via masking, not leakage detection
    leakage_detection_mode: str = "log"  # Default: log-only (no false positive blocks)
    
    # LLM Judge (TEMPORARY - for testing only, disables LLM judge path when True)
    temporarily_disable_llm_judge_path_for_testing: bool = False  # Set to False when LLM judge implementation is complete

    # Cache
    enable_semantic_cache: bool = False
    
    # =========================================================================
    # CLAIMS API RESPONSE CACHE
    # =========================================================================
    # These settings control caching of external Claims API responses.
    # Caching reduces API calls for follow-up questions about the same claim.
    # Cache key format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
    
    # Feature flag to enable/disable caching (set ENABLE_CLAIMS_API_CACHE=false to disable)
    # When disabled, all requests go directly to the external API
    enable_claims_api_cache: bool = True
    
    # Time-To-Live for cached responses in seconds (default: 1 hour)
    # After this time, Redis automatically deletes the cached entry
    # Recommended: 300 (5min) for dev, 1800 (30min) for QA, 3600 (1hr) for prod
    claims_api_cache_ttl_seconds: int = 3600

    # Orchestrator
    remove_punctuation_in_normalization: bool = True

    # Memory Store (cache and session memory)
    memory_store_type: str = "redis"  # Options: "inmemory", "redis", "memorystore"
    memory_store_host: str | None = None
    memory_store_port: int = 6379

    # Redis/Memorystore Configuration (populated by Vault sidecar)
    # DEV: REDIS_HOST=10.236.128.156, REDIS_PORT=6378, REDIS_PASSWORD from Vault
    # PROD: Values injected by Vault sidecar automatically
    redis_host: str = ""  # ⚠️ From Vault/deployment config (REDIS_HOST)
    redis_port: int = 6378  # ⚠️ Overridden by REDIS_PORT env var
    redis_password: str = ""  # ⚠️ From Vault (REDIS_PASSWORD) - NEVER hardcode
    redis_username: str = ""  # ⚠️ From Vault (REDIS_USERNAME) - optional, for Redis ACL
    redis_db: int = 0  # ⚠️ Overridden by REDIS_DB env var
    redis_ssl: bool = True  # ⚠️ Overridden by REDIS_SSL env var - set True in production
    redis_ssl_cert_reqs: str = ""  # Options: "required", "optional", "none"


    # Persistence Store (telemetry and analytics)
    # ⚠️ IMPORTANT: This default is OVERRIDDEN by PERSISTENCE_STORE_TYPE in .env file
    # The .env file takes precedence - set PERSISTENCE_STORE_TYPE=sqlite or PERSISTENCE_STORE_TYPE=mongodb in .env
    # Options:
    #   - "sqlite": Local SQLite database (no server required, good for development)
    #   - "mongodb": MongoDB database (requires MongoDB server, set connection details below)
    #   - "firestore": Google Cloud Firestore (future)
    #   - "bigquery": Google BigQuery (future)
    persistence_store_type: str = "mongodb"  # Default fallback (overridden by .env)
    telemetry_db_path: str = "data/telemetry.db"  # Only used for SQLite
    
    # MongoDB Configuration (only used when persistence_store_type="mongodb")
    # Vault injects: MONGODB_USER, MONGODB_PASSWORD
    # Deployment config injects: MONGODB_HOST, MONGODB_DATABASE_NAME
    mongodb_user: str = ""  # ⚠️ From Vault (MONGODB_USER)
    mongodb_password: str = ""  # ⚠️ From Vault (MONGODB_PASSWORD)
    mongodb_host: str = ""  # ⚠️ From deployment config (MONGODB_HOST)
    mongodb_database_name: str = ""  # ⚠️ From deployment config (MONGODB_DATABASE_NAME)
    
    # Legacy - for local development only (ignored if above env vars are set)
    mongodb_connection_string: str = "mongodb://localhost:27017"

    use_mongodb_for_embeddings: bool = True  # True = MongoDB Vector Search (scalable) | False = .pkl file (local dev)
    
    enable_telemetry: bool = True

    # Checkpoint
    checkpoint_db_path: str = "checkpoints.db"
    enable_checkpointing: bool = True  # Enable to persist state across turns for follow-up questions

    # App
    environment: str = "development"  # ⚠️ Overridden by ENVIRONMENT env var
    debug: bool = True  # ⚠️ Overridden by DEBUG env var

    # API Configuration
    swagger_url: str = "https://claiminquiry-cap-qa.myclaims.pss-np.caremark.com"  # ⚠️ Overridden by SWAGGER_URL env var - Base URL for external APIs
    
    # API Endpoints (can be overridden via environment variables)
    api_endpoint_claim_details: str = "/myclaims/claims/v1/claim/byclaimnumberandseq"  # ⚠️ Overridden by API_ENDPOINT_CLAIM_DETAILS env var
    api_endpoint_claim_list: str = "/myclaims/claims/v1/claim/byclaimnumber"  # ⚠️ Overridden by API_ENDPOINT_CLAIM_LIST env var
    
    # API Fallback Configuration (for testing when external APIs are down)
    enable_api_fallback: bool = True  # ⚠️ Overridden by ENABLE_API_FALLBACK env var - Set to False in production to return real errors
    # When True: Uses mock data if API server is down (5xx errors, timeouts) - allows testing to continue
    # When False: Returns actual errors to users - production behavior
    # Note: Fallback is NEVER used for client errors (400 "claim not found") - those always return errors

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

    # =========================================================================
    # RECOMMENDATION CHIPS CONFIGURATION
    # =========================================================================
    # Generates contextual follow-up suggestions with each response.
    # Recommendations help guide users to logical next steps in their inquiry.
    # When enabled, response_agent returns both response text and recommendation chips.

    # Master switch for recommendation chips feature
    # Set ENABLE_RECOMMENDATIONS=false in .env or environment to disable
    enable_recommendations: bool = True  # ⚠️ Overridden by ENABLE_RECOMMENDATIONS env var

    # Number of recommendation chips to generate per response (1-5)
    # Higher values may increase response latency and token usage
    max_recommendations: int = 2  # ⚠️ Overridden by MAX_RECOMMENDATIONS env var

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
