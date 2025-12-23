from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import Optional
import uuid
from app.analyzers.certificate_analyzer import ProductionCertificateAnalyzer
from app.utils.security import validate_api_key
from app.models.schemas import (
    UploadAnalysisRequest,
    URLAnalysisRequest,
    AnalysisResponse
)
import logging
from typing import List, Form 
logger = logging.getLogger(__name__)
router = APIRouter()
analyzer = ProductionCertificateAnalyzer()

@router.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    provider_id: str = Form(...),
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(validate_api_key)
):
    """Analyze uploaded certificate file"""
    try:
        request_id = f"upl_{uuid.uuid4().hex[:12]}"
        
        # Validate file type
        allowed_types = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ''
        if file_ext not in allowed_types:
            raise HTTPException(400, f"File type not supported. Allowed: {allowed_types}")
        
        # Perform analysis
        report = await analyzer.analyze_certificate(
            source=file,
            provider_id=provider_id,
            request_id=request_id,
            source_type="upload"
        )
        
        return AnalysisResponse(
            request_id=request_id,
            analysis_id=report['analysis_id'],
            status="completed",
            report=report
        )
        
    except Exception as e:
        logger.error(f"Upload analysis failed: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@router.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(
    request: URLAnalysisRequest,
    background_tasks: BackgroundTasks = None,
    api_key: str = Depends(validate_api_key)
):
    """Analyze certificate from URL"""
    try:
        request_id = f"url_{uuid.uuid4().hex[:12]}"
        
        report = await analyzer.analyze_certificate(
            source=request.document_url,
            provider_id=request.provider_id,
            request_id=request_id,
            source_type="url"
        )
        
        return AnalysisResponse(
            request_id=request_id,
            analysis_id=report['analysis_id'],
            status="completed",
            report=report
        )
        
    except Exception as e:
        logger.error(f"URL analysis failed: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@router.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    provider_id: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    """Batch analyze multiple certificates"""
    # Implementation for batch processing
    pass