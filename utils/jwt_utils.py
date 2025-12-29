"""
JWT Token Utility - Extract user information from JWT tokens for audit logging.

This module decodes JWT tokens to extract user details (email, name, etc.)
for compliance audit logging in MongoDB. Signature verification is NOT performed
here as it's handled by the API Gateway upstream.

Usage:
    from utils.jwt_utils import extract_user_info_from_jwt
    
    user_claims = extract_user_info_from_jwt(auth_header)
    # Returns: {"user_email": "...", "user_name": "...", ...}
"""

import base64
import json
from typing import Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """
    Decode JWT payload (claims) without signature verification.
    
    Args:
        token: JWT token string (with or without "Bearer " prefix)
        
    Returns:
        Dict of claims from JWT payload, or empty dict if decoding fails
    """
    try:
        # Handle None or empty
        if not token:
            return {}
            
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        token = token.strip()
        if not token:
            return {}
        
        # JWT structure: header.payload.signature (3 parts separated by dots)
        parts = token.split(".")
        if len(parts) != 3:
            logger.debug(f"Token is not JWT format (expected 3 parts, got {len(parts)})")
            return {}
        
        # Decode payload (second part) - Base64 URL decoding
        payload = parts[1]
        
        # Add padding if needed (Base64 requires length divisible by 4)
        padding_needed = 4 - len(payload) % 4
        if padding_needed != 4:
            payload += "=" * padding_needed
        
        decoded_bytes = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded_bytes.decode("utf-8"))
        
        logger.debug(f"✅ JWT decoded successfully, claims: {list(claims.keys())}")
        return claims
        
    except Exception as e:
        # Don't log as error - token might be opaque token (not JWT)
        logger.debug(f"JWT decode skipped (non-JWT or invalid): {type(e).__name__}")
        return {}


def extract_user_info_from_jwt(auth_token: str) -> Dict[str, Any]:
    """
    Extract all relevant user information from JWT token for audit logging.
    
    This function extracts user details from JWT claims and returns them
    in a format that can be merged into user_info for MongoDB logging.
    
    Extracted fields (if present in JWT):
        - user_email: User's email address (from 'email', 'mail', or 'preferred_username')
        - user_name: User's full name (from 'name')
        - user_first_name: First name (from 'given_name' or 'firstName')
        - user_last_name: Last name (from 'family_name' or 'lastName')
        - jwt_subject: Subject/User ID (from 'sub')
        - jwt_client_id: Client/Application ID (from 'client_id' or 'azp')
        - jwt_issuer: Token issuer (from 'iss')
    
    Args:
        auth_token: Authorization header value (e.g., "Bearer eyJ...")
        
    Returns:
        Dict with extracted user info fields (only includes non-None values)
        
    Example:
        >>> info = extract_user_info_from_jwt("Bearer eyJ...")
        >>> print(info)
        {"user_email": "john@example.com", "user_name": "John Doe", ...}
    """
    claims = _decode_jwt_payload(auth_token)
    
    if not claims:
        return {}
    
    extracted = {}
    
    # Email - Primary requirement for compliance audit
    email = claims.get("email") or claims.get("mail") or claims.get("preferred_username")
    if email:
        extracted["user_email"] = email
    
    # Full name
    name = claims.get("name")
    if name:
        extracted["user_name"] = name
    
    # First name (try multiple common claim names)
    first_name = claims.get("given_name") or claims.get("firstName") or claims.get("first_name")
    if first_name:
        extracted["user_first_name"] = first_name
    
    # Last name (try multiple common claim names)
    last_name = claims.get("family_name") or claims.get("lastName") or claims.get("last_name")
    if last_name:
        extracted["user_last_name"] = last_name
    
    # Subject (standard JWT claim for user identifier)
    sub = claims.get("sub")
    if sub:
        extracted["jwt_subject"] = sub
    
    # Client ID (for application tracking)
    client_id = claims.get("client_id") or claims.get("azp") or claims.get("clientId")
    if client_id:
        extracted["jwt_client_id"] = client_id
    
    # Issuer (for audit trail - know which auth server issued token)
    issuer = claims.get("iss")
    if issuer:
        extracted["jwt_issuer"] = issuer
    
    if extracted:
        logger.info(f"📧 JWT user info extracted: {list(extracted.keys())}")
    
    return extracted

