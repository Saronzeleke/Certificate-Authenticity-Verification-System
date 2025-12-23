from fastapi import HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from app.utils.config import settings

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False 
)

async def validate_api_key(
    api_key: str = Security(api_key_header)
):
    """Production-ready API key validation"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    # Use constant-time comparison to prevent timing attacks
    import hmac
    
    expected_key_bytes = settings.api_key.encode()
    provided_key_bytes = api_key.encode()
    
    if not hmac.compare_digest(expected_key_bytes, provided_key_bytes):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key