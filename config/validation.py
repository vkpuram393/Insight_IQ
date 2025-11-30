"""
Configuration Validation - Startup Checks

Validates critical configuration values at startup to catch misconfigurations early.
This prevents runtime errors from wrong settings (e.g., QA picking up local .env values).
"""

from typing import List, Tuple, Set
from pathlib import Path
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Valid environment names - update this list as needed
# Development environments
DEV_ENVIRONMENTS: Set[str] = {"development", "dev"}

# Upper/non-production environments
UPPER_ENVIRONMENTS: Set[str] = {"qa", "uat", "preprod"}

# Production environments
PROD_ENVIRONMENTS: Set[str] = {"production", "prod"}

# All valid environments
VALID_ENVIRONMENTS: Set[str] = DEV_ENVIRONMENTS | UPPER_ENVIRONMENTS | PROD_ENVIRONMENTS

# Environments that should NOT use dev settings (all except development/dev)
NON_DEV_ENVIRONMENTS: Set[str] = UPPER_ENVIRONMENTS | PROD_ENVIRONMENTS


def validate_environment_config() -> Tuple[bool, List[str]]:
    """
    Validate critical environment-specific configuration values.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    # 1. Validate Environment Variable
    expected_env = settings.environment.lower()
    if expected_env not in VALID_ENVIRONMENTS:
        valid_envs_str = ", ".join(sorted(VALID_ENVIRONMENTS))
        errors.append(
            f"❌ INVALID ENVIRONMENT: '{settings.environment}' is not a valid environment. "
            f"Expected one of: {valid_envs_str}"
        )
    
    # 2. Validate Project ID matches environment
    project_id = settings.project_id.lower()
    env = expected_env
    
    if env in NON_DEV_ENVIRONMENTS:
        # Upper/production environments should NOT use dev project
        if "dev" in project_id and env not in DEV_ENVIRONMENTS:
            errors.append(
                f"❌ CONFIGURATION MISMATCH: Environment is '{env}' but PROJECT_ID contains 'dev': '{settings.project_id}'. "
                f"This suggests local .env values are being used in upper environment!"
            )
        
        # Check for expected project patterns
        if env == "qa" and "qa" not in project_id and "nonprod" not in project_id:
            warnings.append(
                f"⚠️  WARNING: Environment is 'qa' but PROJECT_ID '{settings.project_id}' doesn't contain 'qa' or 'nonprod'"
            )
        elif env == "preprod" and "preprod" not in project_id and "nonprod" not in project_id:
            warnings.append(
                f"⚠️  WARNING: Environment is 'preprod' but PROJECT_ID '{settings.project_id}' doesn't contain 'preprod' or 'nonprod'"
            )
        elif env in PROD_ENVIRONMENTS:
            if "prod" not in project_id and "nonprod" in project_id:
                errors.append(
                    f"❌ CONFIGURATION MISMATCH: Environment is '{env}' but PROJECT_ID '{settings.project_id}' contains 'nonprod'"
                )
    
    # 3. Validate MongoDB Database Name matches environment
    if settings.persistence_store_type == "mongodb":
        db_name = settings.mongodb_database_name.lower()
        
        if env == "qa":
            if "qa" not in db_name and "dev" in db_name:
                errors.append(
                    f"❌ CONFIGURATION MISMATCH: Environment is 'qa' but MONGODB_DATABASE_NAME is '{settings.mongodb_database_name}' (looks like dev). "
                    f"This suggests wrong environment variables are set!"
                )
        elif env == "uat":
            if "uat" not in db_name and "dev" in db_name:
                errors.append(
                    f"❌ CONFIGURATION MISMATCH: Environment is 'uat' but MONGODB_DATABASE_NAME is '{settings.mongodb_database_name}' (looks like dev). "
                    f"This suggests wrong environment variables are set!"
                )
        elif env == "preprod":
            if "preprod" not in db_name and "dev" in db_name:
                errors.append(
                    f"❌ CONFIGURATION MISMATCH: Environment is 'preprod' but MONGODB_DATABASE_NAME is '{settings.mongodb_database_name}' (looks like dev). "
                    f"This suggests wrong environment variables are set!"
                )
        elif env in PROD_ENVIRONMENTS:
            if "pt" not in db_name and "prod" not in db_name:
                errors.append(
                    f"❌ CONFIGURATION MISMATCH: Environment is '{env}' but MONGODB_DATABASE_NAME is '{settings.mongodb_database_name}' (doesn't look like production). "
                    f"This suggests wrong environment variables are set!"
                )
    
    # 4. Validate SWAGGER_URL matches environment
    swagger_url = settings.swagger_url.lower()
    if env == "qa":
        if "qa" not in swagger_url and "dev" in swagger_url:
            warnings.append(
                f"⚠️  WARNING: Environment is 'qa' but SWAGGER_URL '{settings.swagger_url}' contains 'dev' instead of 'qa'"
            )
    elif env == "uat":
        if "uat" not in swagger_url and "dev" in swagger_url:
            warnings.append(
                f"⚠️  WARNING: Environment is 'uat' but SWAGGER_URL '{settings.swagger_url}' contains 'dev' instead of 'uat'"
            )
    elif env == "preprod":
        if "preprod" not in swagger_url and "dev" in swagger_url:
            warnings.append(
                f"⚠️  WARNING: Environment is 'preprod' but SWAGGER_URL '{settings.swagger_url}' contains 'dev' instead of 'preprod'"
            )
    elif env in PROD_ENVIRONMENTS:
        if "prod" not in swagger_url and ("dev" in swagger_url or "qa" in swagger_url or "uat" in swagger_url or "preprod" in swagger_url):
            errors.append(
                f"❌ CONFIGURATION MISMATCH: Environment is '{env}' but SWAGGER_URL '{settings.swagger_url}' points to non-production environment"
            )
    
    # 5. Validate DEBUG flag in upper environments
    if env in NON_DEV_ENVIRONMENTS:
        if settings.debug:
            warnings.append(
                f"⚠️  WARNING: DEBUG is enabled in '{env}' environment. Should be False in upper environments."
            )
    
    # 6. Check for .env file in upper environments (shouldn't exist)
    if env in NON_DEV_ENVIRONMENTS:
        env_file = Path(".env")
        if env_file.exists():
            warnings.append(
                f"⚠️  WARNING: .env file exists in '{env}' environment. "
                f"Upper environments should use environment variables only, not .env files."
            )
    
    # Log all findings
    if errors:
        logger.error("=" * 80)
        logger.error("🚨 CONFIGURATION VALIDATION FAILED")
        logger.error("=" * 80)
        for error in errors:
            logger.error(error)
        logger.error("=" * 80)
        logger.error("Application will fail to start. Please fix configuration errors above.")
        logger.error("=" * 80)
    
    if warnings:
        logger.warning("=" * 80)
        logger.warning("⚠️  CONFIGURATION WARNINGS")
        logger.warning("=" * 80)
        for warning in warnings:
            logger.warning(warning)
        logger.warning("=" * 80)
    
    if not errors and not warnings:
        logger.info("✅ Configuration validation passed")
        logger.info(f"   Environment: {settings.environment}")
        logger.info(f"   Project ID: {settings.project_id}")
        logger.info(f"   Persistence: {settings.persistence_store_type}")
        if settings.persistence_store_type == "mongodb":
            logger.info(f"   MongoDB DB: {settings.mongodb_database_name}")
        logger.info(f"   Debug: {settings.debug}")
    
    return len(errors) == 0, errors


def validate_critical_settings() -> Tuple[bool, List[str]]:
    """
    Validate critical settings that would cause runtime failures.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # 1. Validate persistence store type
    if settings.persistence_store_type not in ["sqlite", "mongodb", "firestore", "bigquery"]:
        errors.append(
            f"❌ INVALID PERSISTENCE_STORE_TYPE: '{settings.persistence_store_type}'. "
            f"Must be one of: sqlite, mongodb, firestore, bigquery"
        )
    
    # 2. Validate MongoDB connection if using MongoDB
    if settings.persistence_store_type == "mongodb":
        if not settings.mongodb_connection_string or settings.mongodb_connection_string == "mongodb://localhost:27017":
            errors.append(
                f"❌ MONGODB_CONNECTION_STRING is not set or is using default localhost. "
                f"Current value: '{settings.mongodb_connection_string}'"
            )
        
        if not settings.mongodb_database_name:
            errors.append(
                f"❌ MONGODB_DATABASE_NAME is not set"
            )
    
    # 3. Validate project_id is set
    if not settings.project_id or settings.project_id == "":
        errors.append(
            f"❌ PROJECT_ID is not set"
        )
    
    # 4. Validate swagger_url is set
    if not settings.swagger_url or settings.swagger_url == "":
        errors.append(
            f"❌ SWAGGER_URL is not set"
        )
    
    if errors:
        logger.error("=" * 80)
        logger.error("🚨 CRITICAL SETTINGS VALIDATION FAILED")
        logger.error("=" * 80)
        for error in errors:
            logger.error(error)
        logger.error("=" * 80)
    
    return len(errors) == 0, errors


def validate_all() -> bool:
    """
    Run all validation checks.
    
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("🔍 Starting configuration validation...")
    
    # Critical settings (must pass)
    critical_valid, critical_errors = validate_critical_settings()
    
    # Environment-specific validation (warnings + errors)
    env_valid, env_errors = validate_environment_config()
    
    all_errors = critical_errors + env_errors
    
    if all_errors:
        logger.error("")
        logger.error("❌ Configuration validation FAILED. Application cannot start safely.")
        logger.error("")
        logger.error("Please fix the following issues:")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  {i}. {error}")
        logger.error("")
        return False
    
    logger.info("✅ All configuration validations passed")
    return True

