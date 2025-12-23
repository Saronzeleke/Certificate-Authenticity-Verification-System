from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional

class AnalysisRequest(BaseModel):
    document_url: HttpUrl
    provider_id: str
    request_id: str

class AnalysisResponse(BaseModel):
    request_id: str
    analysis_id: str
    status: str
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str