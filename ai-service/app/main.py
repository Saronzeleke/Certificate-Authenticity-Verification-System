from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analyzers.certificate_analyzer import ProductionCertificateAnalyzer
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.utils.security import validate_api_key
from app.utils.config import settings
from app.utils.cache import init_redis, close_redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global analyzer instance - initialized in lifespan
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global analyzer
    
    # Startup
    logger.info("Starting ServeEase Certificate Analyzer...")
    
    # Initialize Redis FIRST
    redis_initialized = await init_redis()
    if not redis_initialized:
        logger.warning("⚠️ Redis initialization failed - running in degraded mode")
    
    # Initialize analyzer AFTER Redis
    try:
        analyzer = ProductionCertificateAnalyzer()
        logger.info(f"✅ Analyzer v{analyzer.model_version} ready in {settings.environment} mode")
        logger.info(f"📊 Thresholds - Reject: {settings.reject_threshold}, Low Quality: {settings.low_quality_threshold}")
    except Exception as e:
        logger.error(f"❌ Analyzer initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ServeEase Certificate Analyzer...")
    await close_redis()
    logger.info("✅ Clean shutdown completed")

app = FastAPI(
    title="ServeEase Certificate Analysis API",
    description="AI-powered certificate verification and tampering detection for ServeEase",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "ServeEase Certificate Analyzer",
        "version": "2.1.0",
        "environment": settings.environment,
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "upload": "/api/analyze/upload (POST)",
            "url": "/api/analyze (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        },
        "specs": {
            "max_file_size": f"{settings.max_file_size // (1024*1024)}MB",
            "allowed_extensions": settings.allowed_extensions,
            "reject_threshold": settings.reject_threshold
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with dependency status"""
    try:
        # Check analyzer status
        analyzer_status = "ready" if analyzer else "not_initialized"
        
        # Check Redis status
        from app.utils.cache import redis_pool
        redis_status = "connected" if await redis_pool.health_check() else "disconnected"
        
        # Check analyzer health if available
        analyzer_health = {}
        if analyzer:
            try:
                analyzer_health = await analyzer.health_check()
            except:
                analyzer_health = {"status": "health_check_failed"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "certificate-analyzer",
            "environment": settings.environment,
            "version": "2.1.0",
            "dependencies": {
                "redis": redis_status,
                "analyzer": analyzer_status
            },
            "analyzer_health": analyzer_health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_certificate(
    request: AnalysisRequest,
    api_key: str = Depends(validate_api_key)
):
    """Analyze certificate document from URL"""
    try:
        if not analyzer:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
        logger.info(f"Starting URL analysis for request: {request.request_id}")
        
        # Validate document URL
        if not request.document_url.startswith(('https://', 'http://')):
            raise HTTPException(status_code=400, detail="Invalid document URL. Must start with http:// or https://")
        
        # Validate URL length
        if len(request.document_url) > 2000:
            raise HTTPException(status_code=400, detail="URL too long")
        
        # Validate provider_id
        if not request.provider_id or len(request.provider_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Provider ID is required")
        
        # Perform analysis
        start_time = datetime.now()
        report = await analyzer.analyze_certificate(
            request.document_url,
            request.provider_id,
            request.request_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        report['processing_time'] = round(processing_time, 3)
        
        logger.info(f"URL analysis completed for {request.request_id}. Time: {processing_time}s")
        
        return AnalysisResponse(
            request_id=request.request_id,
            analysis_id=report['analysis_id'],
            status="completed",
            report=report
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error for {request.request_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"URL analysis failed for {request.request_id}: {str(e)}", exc_info=True)
        return AnalysisResponse(
            request_id=request.request_id,
            analysis_id=f"error_{uuid.uuid4().hex[:8]}",
            status="failed",
            error=f"Analysis failed: {str(e)}"
        )

@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    provider_id: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    """Analyze uploaded certificate file"""
    try:
        if not analyzer:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
        request_id = f"upload_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting upload analysis for {request_id}, file: {file.filename}")
        
        # Validate provider_id
        if not provider_id or len(provider_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Provider ID is required")
        
        # Validate file type
        allowed_types = settings.allowed_extensions
        if file.filename:
            file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            if not file_ext or file_ext not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type .{file_ext if file_ext else 'unknown'} not supported. Allowed: {', '.join(allowed_types)}"
                )
        else:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Validate file size
        content = await file.read()
        file_size = len(content)
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Size: {file_size // 1024}KB, Max: {settings.max_file_size // 1024}KB"
            )
        
        # Reset file pointer
        await file.seek(0)
        
        # Perform analysis
        start_time = datetime.now()
        report = await analyzer.analyze_certificate_file(file, provider_id, request_id)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        report['processing_time'] = round(processing_time, 3)
        
        logger.info(f"Upload analysis completed for {request_id}. Time: {processing_time}s")
        
        return {
            "request_id": request_id,
            "analysis_id": report['analysis_id'],
            "status": "completed",
            "report": report
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error for upload: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_result(
    analysis_id: str, 
    api_key: str = Depends(validate_api_key)
):
    """Get analysis result by ID (simplified - in production, fetch from database)"""
    try:
        if not analyzer:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
        # Validate analysis_id format
        if not analysis_id.startswith('anal_') or len(analysis_id) < 10:
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")
        
        # In production, this would fetch from database/cache
        # For now, return a placeholder response
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "note": "Analysis results are only available immediately after processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

@app.get("/api/metrics")
async def get_metrics(api_key: str = Depends(validate_api_key)):
    """Get service metrics (simplified version)"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "service": "certificate-analyzer",
            "version": "2.1.0",
            "environment": settings.environment,
            "uptime": "N/A",  # Would require uptime tracking
            "requests_processed": "N/A",  # Would require request counting
            "average_processing_time": "N/A",
            "redis_status": "N/A"
        }
        
        # Add Redis metrics if available
        try:
            from app.utils.cache import redis_pool
            redis_health = await redis_pool.health_check()
            metrics["redis_status"] = "healthy" if redis_health else "unhealthy"
        except:
            metrics["redis_status"] = "unknown"
        
        return metrics
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics unavailable: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info",
        access_log=True
    )