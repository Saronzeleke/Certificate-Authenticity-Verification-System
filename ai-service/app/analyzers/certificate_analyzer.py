import logging
import hashlib
import tempfile
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import time

from app.analyzers.tamper_detector import ProductionTamperDetector
from app.analyzers.ocr_engine import ProductionOCREngine
from app.utils.image_processing import ImageProcessor
from fastapi import UploadFile
import aiohttp
from app.utils.config import settings
import aiofiles
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ProductionCertificateAnalyzer:
    """Production-ready certificate analyzer supporting both uploads and URLs with Redis safety"""

    def __init__(self):
        self.tamper_detector = ProductionTamperDetector()
        self.ocr_engine = ProductionOCREngine()
        self.image_processor = ImageProcessor()
        self.model_version = "2024.1.0-prod"
        
        # Configurable thresholds
        self.reject_threshold = float(settings.reject_threshold)
        self.low_quality_threshold = 0.7
        self.CACHE_TTL = 3600
        
        # Don't initialize Redis here - it's already done in main.py
        self.config_hash = self._get_config_hash()
        
        # Redis safety flag
        self._redis_available = True
        
        logger.info(f"ProductionCertificateAnalyzer v{self.model_version} initialized")

    def _get_config_hash(self) -> str:
        """Generate config hash for version tracking"""
        config_data = {
            'tamper_model': self.tamper_detector.model_version,
            'ocr_version': self.ocr_engine.model_version,
            'weights': self.ocr_engine.AUTHENTICITY_WEIGHTS,
            'reject_threshold': self.reject_threshold
        }
        return hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:16]
    
    async def _get_redis_client(self):
        """Safely get Redis client with fallback"""
        try:
            from app.utils.cache import get_redis_client
            client = await get_redis_client()
            return client
        except ImportError:
            logger.error("Redis module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to get Redis client: {e}")
            return None
    
    async def _safe_redis_setex(self, cache_key: str, ttl: int, data: dict) -> bool:
        """Safely set cache with fallback"""
        try:
            client = await self._get_redis_client()
            if client:
                await client.setex(cache_key, ttl, json.dumps(data))
                logger.debug(f"Cache set for key: {cache_key}")
                return True
            else:
                logger.warning("Redis not available, skipping cache set")
                self._redis_available = False
                return False
        except Exception as e:
            logger.error(f"Redis setex failed: {e}")
            self._redis_available = False
            return False
    
    async def _safe_redis_get(self, cache_key: str) -> Optional[dict]:
        """Safely get from cache with fallback"""
        try:
            client = await self._get_redis_client()
            if client:
                cached = await client.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    async def _get_file_hash(self, file: UploadFile) -> str:
        """Generate hash for uploaded file"""
        try:
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate file hash: {e}")
            return "unknown"
    
    async def analyze_certificate_file(
        self, 
        file: UploadFile, 
        provider_id: str, 
        request_id: str
    ) -> Dict[str, Any]:
        """Analyze uploaded certificate file with Redis safety"""
        temp_path = None
        try:
            logger.info(f"Starting file upload analysis for {request_id}, file: {file.filename}")
            start_time = datetime.now()
            
            # Check cache first (if Redis is available)
            file_hash = await self._get_file_hash(file)
            await file.seek(0)  # Reset again after hash
            cache_key = f"upload:{provider_id}:{file_hash}"
            
            if self._redis_available:
                cached_result = await self._safe_redis_get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for upload analysis {request_id}")
                    cached_result['cache_hit'] = True
                    cached_result['redis_available'] = self._redis_available
                    return cached_result
            
            # Save uploaded file
            temp_path = await self._save_upload(file)
            
            # Process document
            processed_images = await self.image_processor.process_document(temp_path)
            if not processed_images:
                raise Exception("No valid images extracted from document")
            
            logger.info(f"Processed {len(processed_images)} images")
            
            # Parallel analysis of pages
            page_tasks = [
                self._analyze_single_page(img, idx, request_id)
                for idx, img in enumerate(processed_images)
            ]
            analysis_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
            # Handle any failed pages
            valid_results = []
            for idx, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    logger.error(f"Page {idx} analysis failed: {str(result)}")
                else:
                    valid_results.append(result)
            
            if not valid_results:
                raise Exception("All page analyses failed")
            
            # Combine results
            final_report = self._combine_analysis_results(
                valid_results, provider_id, request_id
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_report['processing_time'] = round(processing_time, 3)
            final_report['model_version'] = self.model_version
            final_report['config_hash'] = self.config_hash
            final_report['source_type'] = "upload"
            final_report['cache_hit'] = False
            final_report['redis_available'] = self._redis_available
            
            # Cache result (only if Redis is available)
            if self._redis_available:
                cache_success = await self._safe_redis_setex(
                    cache_key, 
                    self.CACHE_TTL, 
                    final_report
                )
                final_report['cache_stored'] = cache_success
            else:
                final_report['cache_stored'] = False
            
            # Store training data (async, non-blocking)
            try:
                asyncio.create_task(
                    self._store_training_data(valid_results, final_report, "upload")
                )
            except Exception as e:
                logger.error(f"Failed to schedule training data storage: {e}")
            
            logger.info(f"Upload analysis completed in {processing_time}s. Score: {final_report['authenticity_score']}")
            return final_report
            
        except Exception as e:
            logger.error(f"File analysis pipeline failed: {str(e)}", exc_info=True)
            error_report = self._generate_error_report(
                provider_id, request_id, str(e), "upload"
            )
            return error_report
        finally:
            # Cleanup
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {str(e)}")
    
    async def analyze_certificate(
        self, 
        document_url: str, 
        provider_id: str, 
        request_id: str
    ) -> Dict[str, Any]:
        """Analyze certificate from URL with Redis safety"""
        temp_path = None
        try:
            logger.info(f"Starting URL analysis for {request_id}, URL: {document_url[:100]}...")
            start_time = datetime.now()
            
            # Check cache first
            cache_key = f"url:{provider_id}:{hashlib.md5(document_url.encode()).hexdigest()}"
            
            if self._redis_available:
                cached_result = await self._safe_redis_get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for {request_id}")
                    cached_result['cache_hit'] = True
                    cached_result['redis_available'] = self._redis_available
                    return cached_result
            
            # Download document
            temp_path = await self._download_document(document_url)
            logger.info(f"Document downloaded to {temp_path}")
            
            # Process document
            processed_images = await self.image_processor.process_document(temp_path)
            if not processed_images:
                raise Exception("No valid images extracted from document")
            
            logger.info(f"Processed {len(processed_images)} images")
            
            # Parallel analysis of pages
            page_tasks = [
                self._analyze_single_page(img, idx, request_id)
                for idx, img in enumerate(processed_images)
            ]
            analysis_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
            # Handle any failed pages
            valid_results = []
            for idx, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    logger.error(f"Page {idx} analysis failed: {str(result)}")
                else:
                    valid_results.append(result)
            
            if not valid_results:
                raise Exception("All page analyses failed")
            
            # Combine results
            final_report = self._combine_analysis_results(
                valid_results, provider_id, request_id
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_report['processing_time'] = round(processing_time, 3)
            final_report['model_version'] = self.model_version
            final_report['config_hash'] = self.config_hash
            final_report['source_type'] = "url"
            final_report['cache_hit'] = False
            final_report['redis_available'] = self._redis_available
            
            # Cache result (only if Redis is available)
            if self._redis_available:
                cache_success = await self._safe_redis_setex(
                    cache_key, 
                    self.CACHE_TTL, 
                    final_report
                )
                final_report['cache_stored'] = cache_success
            else:
                final_report['cache_stored'] = False
            
            # Store training data (async, non-blocking)
            try:
                asyncio.create_task(
                    self._store_training_data(valid_results, final_report, "url")
                )
            except Exception as e:
                logger.error(f"Failed to schedule training data storage: {e}")
            
            logger.info(f"URL analysis completed in {processing_time}s. Score: {final_report['authenticity_score']}")
            return final_report
            
        except Exception as e:
            logger.error(f"URL analysis pipeline failed: {str(e)}", exc_info=True)
            error_report = self._generate_error_report(
                provider_id, request_id, str(e), "url"
            )
            return error_report
        finally:
            # Cleanup
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {str(e)}")
    
    def _generate_error_report(
        self, 
        provider_id: str, 
        request_id: str, 
        error_message: str, 
        source_type: str
    ) -> Dict[str, Any]:
        """Generate error report when analysis fails"""
        return {
            'analysis_id': f"error_{hashlib.md5(f'{provider_id}{request_id}'.encode()).hexdigest()[:12]}",
            'timestamp': datetime.now().isoformat(),
            'provider_id': provider_id,
            'request_id': request_id,
            'authenticity_score': 0.0,
            'status': 'Analysis Failed',
            'error': error_message,
            'source_type': source_type,
            'cache_hit': False,
            'redis_available': self._redis_available,
            'quality_metrics': {
                'ocr_confidence': 0.0,
                'tampering_confidence': 0.0,
                'page_count': 0,
                'average_entropy': 0.0
            },
            'flags': {
                'analysis_failed': True,
                'tampering_detected': False,
                'low_quality_scan': True,
                'incomplete_fields': True
            },
            'model_metadata': {
                'version': self.model_version,
                'config_hash': self.config_hash,
                'thresholds': {
                    'reject': self.reject_threshold,
                    'low_quality': self.low_quality_threshold
                }
            },
            'recommendations': ['Analysis failed. Please try again or contact support.']
        }
    
    async def _save_upload(self, file: UploadFile) -> str:
        """Save UploadFile to temporary file with proper cleanup"""
        try:
            suffix = Path(file.filename).suffix if file.filename else ".pdf"
            if suffix.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                suffix = '.pdf'  # Default to PDF for documents
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                return tmp.name
                
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise Exception(f"File upload failed: {str(e)}")
    
    async def _download_document(self, document_url: str) -> str:
        """Download document from URL with timeout and validation"""
        try:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: Failed to download")
                    
                    # Get file extension from Content-Type or URL
                    content_type = response.headers.get('Content-Type', '')
                    if 'pdf' in content_type.lower():
                        suffix = '.pdf'
                    elif 'image' in content_type.lower():
                        suffix = '.jpg'
                    else:
                        # Extract from URL
                        suffix = Path(document_url).suffix or '.pdf'
                    
                    fd, temp_path = tempfile.mkstemp(suffix=suffix)
                    os.close(fd)
                    
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    return temp_path
                    
        except asyncio.TimeoutError:
            raise Exception("Download timeout - server took too long to respond")
        except Exception as e:
            logger.error(f"Document download failed: {str(e)}")
            raise Exception(f"Download failed: {str(e)}")
    
    async def _analyze_single_page(
        self, 
        image, 
        page_num: int, 
        request_id: str
    ) -> Dict[str, Any]:
        """Analyze single page with comprehensive feature extraction"""
        try:
            # Parallel OCR and tampering detection
            ocr_task = self.ocr_engine.extract_text(image)
            tamper_task = self.tamper_detector.detect_tampering(image)
            
            ocr_results, tampering_results = await asyncio.gather(
                ocr_task, tamper_task, 
                return_exceptions=True
            )
            
            # Handle exceptions in results
            if isinstance(ocr_results, Exception):
                logger.error(f"OCR failed for page {page_num}: {ocr_results}")
                ocr_results = {'error': str(ocr_results), 'ocr_confidence': 0.0}
            
            if isinstance(tampering_results, Exception):
                logger.error(f"Tampering detection failed for page {page_num}: {tampering_results}")
                tampering_results = {'error': str(tampering_results), 'tampering_confidence': 0.0}
            
            # Extract features for ML
            ml_features = self._extract_ml_features(image, ocr_results, tampering_results)
            
            return {
                'page_number': page_num,
                'ocr_results': ocr_results,
                'tampering_results': tampering_results,
                'ml_features': ml_features,
                'extracted_fields': ocr_results.get('extracted_text', {}) if isinstance(ocr_results, dict) else {},
                'field_confidence': ocr_results.get('field_confidence', {}) if isinstance(ocr_results, dict) else {},
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id
            }
            
        except Exception as e:
            logger.error(f"Page {page_num} analysis failed: {str(e)}")
            return {
                'page_number': page_num,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id
            }
    
    def _extract_ml_features(self, image, ocr_results, tampering_results) -> Dict[str, Any]:
        """Extract features for model training/improvement"""
        try:
            image_entropy = self._calculate_image_entropy(image)
        except Exception:
            image_entropy = 0.0
        
        # Safely get values with defaults
        ocr_confidence = 0.0
        language = 'unknown'
        text_length = 0
        
        if isinstance(ocr_results, dict):
            ocr_confidence = ocr_results.get('ocr_confidence', 0.0)
            language = ocr_results.get('language_used', 'unknown')
            text_length = len(ocr_results.get('extracted_text', {}))
        
        ela_score = 0.0
        noise_score = 0.0
        anomaly_score = 0.0
        
        if isinstance(tampering_results, dict):
            ela_score = tampering_results.get('ela_score', 0.0)
            noise_score = tampering_results.get('noise_consistency', 0.0)
            anomaly_score = tampering_results.get('hf_anomaly_score', 0.0)
        
        return {
            'image_stats': {
                'shape': image.shape if hasattr(image, 'shape') else (0, 0, 0),
                'mean_intensity': float(np.mean(image)) if hasattr(image, 'size') and image.size > 0 else 0.0,
                'std_intensity': float(np.std(image)) if hasattr(image, 'size') and image.size > 0 else 0.0,
                'entropy': image_entropy
            },
            'ocr_features': {
                'confidence': ocr_confidence,
                'text_length': text_length,
                'language': language
            },
            'tampering_features': {
                'ela_score': ela_score,
                'noise_score': noise_score,
                'anomaly_score': anomaly_score
            }
        }
    
    def _calculate_image_entropy(self, image) -> float:
        """Calculate image entropy for quality assessment"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            if gray.size == 0:
                return 0.0
                
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return float(entropy)
        except Exception:
            return 0.0
    
    def _combine_analysis_results(
        self, 
        page_results: List[Dict[str, Any]], 
        provider_id: str, 
        request_id: str
    ) -> Dict[str, Any]:
        """Combine multiple page results into final report"""
        if not page_results:
            raise Exception("No analysis results to combine")
        
        # Filter out error pages
        valid_pages = [page for page in page_results if 'error' not in page]
        if not valid_pages:
            # All pages had errors, use first page for error info
            raise Exception(f"All pages failed: {page_results[0].get('error', 'Unknown error')}")
        
        # Use first valid page as primary
        primary_result = valid_pages[0]
        
        # Calculate scores with configurable weights
        authenticity_score = self._calculate_authenticity_score(primary_result)
        
        # Determine status with configurable threshold
        if authenticity_score >= self.reject_threshold:
            status = "Pending Admin Review"
            admin_action = "approve"
        else:
            status = "Auto-Rejected"
            admin_action = "reject"
        
        # Generate secure analysis ID
        analysis_id = self._generate_analysis_id(provider_id, request_id)
        
        # Calculate average entropy safely
        entropies = []
        for page in valid_pages:
            try:
                if 'ml_features' in page and 'image_stats' in page['ml_features']:
                    entropy = page['ml_features']['image_stats']['entropy']
                    entropies.append(entropy)
            except (KeyError, TypeError, AttributeError):
                continue
        avg_entropy = np.mean(entropies) if entropies else 0.0
        
        # Get OCR confidence safely
        ocr_confidence = 0.0
        if 'ocr_results' in primary_result and isinstance(primary_result['ocr_results'], dict):
            ocr_confidence = primary_result['ocr_results'].get('ocr_confidence', 0.0)
        
        # Get tampering confidence safely
        tampering_confidence = 0.0
        if 'tampering_results' in primary_result and isinstance(primary_result['tampering_results'], dict):
            tampering_confidence = primary_result['tampering_results'].get('tampering_confidence', 0.0)
        
        # Build comprehensive report
        report = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'provider_id': provider_id,
            'request_id': request_id,
            'extracted_data': primary_result.get('extracted_fields', {}),
            'confidence_scores': primary_result.get('field_confidence', {}),
            'authenticity_score': round(authenticity_score, 4),
            'status': status,
            'admin_action': admin_action,
            'quality_metrics': {
                'ocr_confidence': ocr_confidence,
                'tampering_confidence': tampering_confidence,
                'page_count': len(page_results),
                'valid_page_count': len(valid_pages),
                'average_entropy': round(avg_entropy, 4),
                'failed_pages': len(page_results) - len(valid_pages)
            },
            'flags': self._generate_flags(primary_result, valid_pages),
            'page_summaries': [
                {
                    'page': page['page_number'],
                    'tampering_detected': page.get('tampering_results', {}).get('tampering_detected', False) if isinstance(page.get('tampering_results'), dict) else False,
                    'ocr_confidence': page.get('ocr_results', {}).get('ocr_confidence', 0.0) if isinstance(page.get('ocr_results'), dict) else 0.0,
                    'status': 'success' if 'error' not in page else 'failed'
                }
                for page in page_results
            ],
            'model_metadata': {
                'version': self.model_version,
                'config_hash': self.config_hash,
                'thresholds': {
                    'reject': self.reject_threshold,
                    'low_quality': self.low_quality_threshold
                }
            },
            'recommendations': self._generate_recommendations(authenticity_score, primary_result)
        }
        
        return report
    
    def _calculate_authenticity_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate authenticity score with configurable weights"""
        try:
            weights = self.ocr_engine.AUTHENTICITY_WEIGHTS
            
            # Get OCR confidence safely
            ocr_confidence = 0.0
            if 'ocr_results' in analysis_result and isinstance(analysis_result['ocr_results'], dict):
                ocr_confidence = analysis_result['ocr_results'].get('ocr_confidence', 0.0)
            
            # Get tampering confidence safely
            tampering_confidence = 0.0
            if 'tampering_results' in analysis_result and isinstance(analysis_result['tampering_results'], dict):
                tampering_confidence = 1.0 - analysis_result['tampering_results'].get('tampering_confidence', 0.0)
            
            # Field completeness based on expected fields
            extracted_fields = analysis_result.get('extracted_fields', {})
            expected_fields = self.ocr_engine.get_expected_fields()
            present_fields = [f for f in expected_fields if extracted_fields.get(f)]
            field_completeness = len(present_fields) / len(expected_fields) if expected_fields else 1.0
            
            # Calculate weighted score
            score = (
                weights['ocr_quality'] * ocr_confidence +
                weights['tampering'] * tampering_confidence +
                weights['field_completeness'] * field_completeness
            )
            
            # Apply sigmoid for non-linearity
            score = 1 / (1 + np.exp(-10 * (score - 0.5)))
            
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Authenticity score calculation failed: {e}")
            return 0.0
    
    def _generate_flags(self, primary_result, all_results):
        """Generate quality and integrity flags"""
        try:
            tampering_detected = False
            if 'tampering_results' in primary_result and isinstance(primary_result['tampering_results'], dict):
                tampering_detected = primary_result['tampering_results'].get('tampering_detected', False)
            
            ocr_confidence = 0.0
            if 'ocr_results' in primary_result and isinstance(primary_result['ocr_results'], dict):
                ocr_confidence = primary_result['ocr_results'].get('ocr_confidence', 0.0)
            
            flags = {
                'tampering_detected': tampering_detected,
                'low_quality_scan': ocr_confidence < self.low_quality_threshold,
                'incomplete_fields': len([f for f in primary_result.get('extracted_fields', {}).values() if f]) < 2,
                'multiple_pages': len(all_results) > 1,
                'suspicious_patterns': self._check_suspicious_patterns(primary_result.get('extracted_fields', {})),
                'expired_certificate': self._check_expiry(primary_result.get('extracted_fields', {}).get('expiry_date'))
            }
            return flags
        except Exception as e:
            logger.error(f"Flag generation failed: {e}")
            return {
                'tampering_detected': False,
                'low_quality_scan': True,
                'incomplete_fields': True,
                'multiple_pages': False,
                'suspicious_patterns': False,
                'expired_certificate': False
            }
    
    def _check_suspicious_patterns(self, fields: Dict[str, str]) -> bool:
        """Check for suspicious patterns in extracted data"""
        if not fields:
            return False
        
        suspicious_indicators = [
            'test', 'example', 'sample', 'fake', 'dummy',
            '@@', '##', '&&', 'XXXX', '00000', '123456'
        ]
        
        for field_value in fields.values():
            if field_value and isinstance(field_value, str):
                field_lower = field_value.lower()
                if any(indicator in field_lower for indicator in suspicious_indicators):
                    return True
        return False
    
    def _check_expiry(self, expiry_date_str: Optional[str]) -> bool:
        """Check if certificate is expired"""
        if not expiry_date_str or not isinstance(expiry_date_str, str):
            return False
        
        try:
            # Try multiple date formats
            import re
            from datetime import datetime
            
            date_patterns = [
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})',  
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, expiry_date_str)
                if match:
                    try:
                        parts = match.groups()
                        if len(parts) == 3:
                            if len(parts[2]) == 4:  
                                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                            elif len(parts[0]) == 4:  
                                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            else:  
                                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                                year += 2000 if year < 100 else 0
                            
                            expiry_date = datetime(year, month, day).date()
                            return expiry_date < datetime.now().date()
                    except (ValueError, IndexError):
                        continue
            return False
        except Exception:
            return False
    
    def _generate_analysis_id(self, provider_id: str, request_id: str) -> str:
        """Generate secure analysis ID"""
        # Use analysis_id_salt from settings or default
        salt = getattr(settings, 'analysis_id_salt', 'serveease-cert-salt-2024')
        hash_input = f"{provider_id}{request_id}{salt}{datetime.now().isoformat()}"
        return f"anal_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"
    
    def _generate_recommendations(self, score: float, analysis_result: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if score < 0.3:
            recommendations.append("Strongly recommend manual verification")
            recommendations.append("Consider requesting original document")
        
        tampering_detected = False
        if 'tampering_results' in analysis_result and isinstance(analysis_result['tampering_results'], dict):
            tampering_detected = analysis_result['tampering_results'].get('tampering_detected', False)
        
        if tampering_detected:
            recommendations.append("Potential tampering detected - escalate to fraud team")
        
        ocr_confidence = 0.0
        if 'ocr_results' in analysis_result and isinstance(analysis_result['ocr_results'], dict):
            ocr_confidence = analysis_result['ocr_results'].get('ocr_confidence', 0.0)
        
        if ocr_confidence < 0.6:
            recommendations.append("Low OCR quality - request clearer scan")
        
        if not recommendations:
            recommendations.append("Proceed with standard verification")
        
        return recommendations
    
    async def _store_training_data(self, page_results: List[Dict], final_report: Dict, source_type: str):
        """Asynchronously store data for model improvement"""
        try:
            # Filter out pages with errors
            valid_pages = [page for page in page_results if 'error' not in page]
            if not valid_pages:
                logger.warning("No valid pages for training data storage")
                return
            
            training_data = {
                'page_results': valid_pages,
                'final_report': final_report,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'source_type': source_type,
                'config_hash': self.config_hash,
                'redis_available': self._redis_available
            }
            
            # Save to local file for now (in production, send to data lake/S3)
            training_dir = Path("training_data")
            training_dir.mkdir(exist_ok=True)
            
            filename = training_dir / f"train_{final_report['analysis_id']}_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)
            
            logger.info(f"Training data saved to {filename} for {final_report['analysis_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store training data: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the analyzer"""
        try:
            # Check Redis availability
            redis_status = "available" if self._redis_available else "unavailable"
            
            # Check if OCR engine is ready
            ocr_ready = hasattr(self.ocr_engine, 'model_version')
            
            # Check if tamper detector is ready
            tamper_ready = hasattr(self.tamper_detector, 'model_version')
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'redis': redis_status,
                    'ocr_engine': 'ready' if ocr_ready else 'not_ready',
                    'tamper_detector': 'ready' if tamper_ready else 'not_ready',
                    'image_processor': 'ready' if hasattr(self.image_processor, 'process_document') else 'not_ready'
                },
                'model_version': self.model_version,
                'config_hash': self.config_hash
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
# import logging
# import hashlib
# import tempfile
# import asyncio
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Union
# from pathlib import Path
# import json

# from app.analyzers.tamper_detector import ProductionTamperDetector
# from app.analyzers.ocr_engine import ProductionOCREngine
# from app.utils.image_processing import ImageProcessor
# from fastapi import UploadFile
# import aiohttp
# from app.utils.config import settings
# from app.utils.cache import redis_client
# import aiofiles
# import numpy as np
# import cv2
# logger = logging.getLogger(__name__)

# class ProductionCertificateAnalyzer:
#     """Production-ready certificate analyzer supporting both uploads and URLs"""
    
#     def __init__(self):
#         self.tamper_detector = ProductionTamperDetector()
#         self.ocr_engine = ProductionOCREngine()
#         self.image_processor = ImageProcessor()
#         self.model_version = "2024.1.0-prod"
#         self.config_hash = self._get_config_hash()
        
#         # Configurable thresholds
#         self.REJECT_THRESHOLD = float(settings.AUTH_REJECT_THRESHOLD)  # From config
#         self.LOW_QUALITY_THRESHOLD = 0.7
#         self.CACHE_TTL = 3600  # 1 hour cache for same certificates
        
#         logger.info(f"ProductionCertificateAnalyzer v{self.model_version} initialized")
    
#     def _get_config_hash(self) -> str:
#         """Generate config hash for version tracking"""
#         config_data = {
#             'tamper_model': self.tamper_detector.model_version,
#             'ocr_version': self.ocr_engine.model_version,
#             'weights': self.ocr_engine.AUTHENTICITY_WEIGHTS
#         }
#         return hashlib.sha256(json.dumps(config_data).encode()).hexdigest()[:16]
    
#     async def analyze_certificate(
#         self, 
#         source: Union[str, UploadFile],  # Can be URL or UploadFile
#         provider_id: str, 
#         request_id: str,
#         source_type: str = "upload"  # "upload" or "url"
#     ) -> Dict[str, Any]:
#         """Main analysis pipeline supporting both uploads and URLs"""
        
#         # Check cache first
#         cache_key = self._generate_cache_key(source, provider_id, source_type)
#         cached_result = await redis_client.get(cache_key)
#         if cached_result:
#             logger.info(f"Cache hit for {request_id}")
#             result = json.loads(cached_result)
#             result['cache_hit'] = True
#             return result
        
#         temp_path = None
#         try:
#             logger.info(f"Starting {source_type} analysis for {request_id}")
#             start_time = datetime.now()
            
#             # Step 1: Get document based on source type
#             if source_type == "upload":
#                 if not isinstance(source, UploadFile):
#                     raise ValueError("Source must be UploadFile for upload type")
#                 temp_path = await self._save_upload(source)
#             elif source_type == "url":
#                 if not isinstance(source, str):
#                     raise ValueError("Source must be string URL for url type")
#                 temp_path = await self._download_document(source)
#             else:
#                 raise ValueError(f"Invalid source_type: {source_type}")
            
#             logger.info(f"Document ready at {temp_path}")
            
#             # Step 2: Process document
#             processed_images = await self.image_processor.process_document(temp_path)
#             if not processed_images:
#                 raise Exception("No valid images extracted from document")
            
#             logger.info(f"Processed {len(processed_images)} images")
            
#             # Step 3: Parallel analysis of pages
#             page_tasks = [
#                 self._analyze_single_page(img, idx, request_id)
#                 for idx, img in enumerate(processed_images)
#             ]
#             analysis_results = await asyncio.gather(*page_tasks, return_exceptions=True)
            
#             # Handle any failed pages
#             valid_results = []
#             for idx, result in enumerate(analysis_results):
#                 if isinstance(result, Exception):
#                     logger.error(f"Page {idx} analysis failed: {str(result)}")
#                 else:
#                     valid_results.append(result)
            
#             if not valid_results:
#                 raise Exception("All page analyses failed")
            
#             # Step 4: Combine results
#             final_report = self._combine_analysis_results(
#                 valid_results, provider_id, request_id
#             )
            
#             # Calculate processing time
#             processing_time = (datetime.now() - start_time).total_seconds()
#             final_report['processing_time'] = round(processing_time, 3)
#             final_report['model_version'] = self.model_version
#             final_report['config_hash'] = self.config_hash
#             final_report['source_type'] = source_type
            
#             # Step 5: Cache result
#             await redis_client.setex(
#                 cache_key, 
#                 self.CACHE_TTL, 
#                 json.dumps(final_report)
#             )
            
#             # Step 6: Store training data (async - don't await)
#             asyncio.create_task(
#                 self._store_training_data(valid_results, final_report)
#             )
            
#             logger.info(f"Analysis completed in {processing_time}s. Score: {final_report['authenticity_score']}")
#             return final_report
            
#         except Exception as e:
#             logger.error(f"Analysis pipeline failed: {str(e)}", exc_info=True)
#             raise
#         finally:
#             # Cleanup
#             if temp_path and Path(temp_path).exists():
#                 try:
#                     Path(temp_path).unlink()
#                     logger.debug(f"Cleaned up temporary file: {temp_path}")
#                 except Exception as e:
#                     logger.warning(f"Failed to clean up temp file: {str(e)}")
    
#     async def _save_upload(self, file: UploadFile) -> str:
#         """Save UploadFile to temporary file with proper cleanup"""
#         try:
#             suffix = Path(file.filename).suffix if file.filename else ".pdf"
#             if suffix.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
#                 suffix = '.pdf'  # Default to PDF for documents
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 content = await file.read()
#                 tmp.write(content)
#                 return tmp.name
                
#         except Exception as e:
#             logger.error(f"Failed to save uploaded file: {str(e)}")
#             raise Exception(f"File upload failed: {str(e)}")
    
#     async def _download_document(self, document_url: str) -> str:
#         """Download document from URL with timeout and validation"""
#         try:
#             timeout = aiohttp.ClientTimeout(total=30, connect=10)
#             async with aiohttp.ClientSession(timeout=timeout) as session:
#                 async with session.get(document_url) as response:
#                     if response.status != 200:
#                         raise Exception(f"HTTP {response.status}: Failed to download")
                    
#                     # Get file extension from Content-Type or URL
#                     content_type = response.headers.get('Content-Type', '')
#                     if 'pdf' in content_type.lower():
#                         suffix = '.pdf'
#                     elif 'image' in content_type.lower():
#                         suffix = '.jpg'
#                     else:
#                         # Extract from URL
#                         suffix = Path(document_url).suffix or '.pdf'
                    
#                     fd, temp_path = tempfile.mkstemp(suffix=suffix)
#                     os.close(fd)
                    
#                     async with aiofiles.open(temp_path, 'wb') as f:
#                         async for chunk in response.content.iter_chunked(8192):
#                             await f.write(chunk)
                    
#                     return temp_path
                    
#         except asyncio.TimeoutError:
#             raise Exception("Download timeout - server took too long to respond")
#         except Exception as e:
#             logger.error(f"Document download failed: {str(e)}")
#             raise Exception(f"Download failed: {str(e)}")
    
#     async def _analyze_single_page(
#         self, 
#         image, 
#         page_num: int, 
#         request_id: str
#     ) -> Dict[str, Any]:
#         """Analyze single page with comprehensive feature extraction"""
#         try:
#             # Parallel OCR and tampering detection
#             ocr_task = self.ocr_engine.extract_text(image)
#             tamper_task = self.tamper_detector.detect_tampering(image)
            
#             ocr_results, tampering_results = await asyncio.gather(
#                 ocr_task, tamper_task
#             )
            
#             # Extract features for ML
#             ml_features = self._extract_ml_features(image, ocr_results, tampering_results)
            
#             return {
#                 'page_number': page_num,
#                 'ocr_results': ocr_results,
#                 'tampering_results': tampering_results,
#                 'ml_features': ml_features,
#                 'extracted_fields': ocr_results['extracted_text'],
#                 'field_confidence': ocr_results.get('field_confidence', {}),
#                 'timestamp': datetime.now().isoformat(),
#                 'request_id': request_id
#             }
            
#         except Exception as e:
#             logger.error(f"Page {page_num} analysis failed: {str(e)}")
#             raise
    
#     def _extract_ml_features(self, image, ocr_results, tampering_results) -> Dict[str, Any]:
#         """Extract features for model training/improvement"""
#         return {
#             'image_stats': {
#                 'shape': image.shape,
#                 'mean_intensity': float(np.mean(image)),
#                 'std_intensity': float(np.std(image)),
#                 'entropy': self._calculate_image_entropy(image)
#             },
#             'ocr_features': {
#                 'confidence': ocr_results.get('ocr_confidence', 0.0),
#                 'text_length': len(ocr_results.get('extracted_text', {})),
#                 'language': ocr_results.get('language_used', 'unknown')
#             },
#             'tampering_features': {
#                 'ela_score': tampering_results.get('ela_score', 0.0),
#                 'noise_score': tampering_results.get('noise_consistency', 0.0),
#                 'anomaly_score': tampering_results.get('hf_anomaly_score', 0.0)
#             }
#         }
    
#     def _calculate_image_entropy(self, image) -> float:
#         """Calculate image entropy for quality assessment"""
#         try:
#             if len(image.shape) == 3:
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = image
            
#             # Calculate histogram
#             hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#             hist = hist.ravel() / hist.sum()
            
#             # Calculate entropy
#             entropy = -np.sum(hist * np.log2(hist + 1e-10))
#             return float(entropy)
#         except:
#             return 0.0
    
#     def _combine_analysis_results(
#         self, 
#         page_results: List[Dict[str, Any]], 
#         provider_id: str, 
#         request_id: str
#     ) -> Dict[str, Any]:
#         """Combine multiple page results into final report"""
#         if not page_results:
#             raise Exception("No analysis results to combine")
        
#         # Use first page as primary (most certificates are single page)
#         primary_result = page_results[0]
        
#         # Calculate scores with configurable weights
#         authenticity_score = self._calculate_authenticity_score(primary_result)
        
#         # Determine status with configurable threshold
#         if authenticity_score >= self.REJECT_THRESHOLD:
#             status = "Pending Admin Review"
#             admin_action = "approve"
#         else:
#             status = "Auto-Rejected"
#             admin_action = "reject"
        
#         # Generate secure analysis ID
#         analysis_id = self._generate_analysis_id(provider_id, request_id)
        
#         # Build comprehensive report
#         report = {
#             'analysis_id': analysis_id,
#             'timestamp': datetime.now().isoformat(),
#             'provider_id': provider_id,
#             'request_id': request_id,
#             'extracted_data': primary_result['extracted_fields'],
#             'confidence_scores': primary_result['field_confidence'],
#             'authenticity_score': round(authenticity_score, 4),
#             'status': status,
#             'admin_action': admin_action,
#             'quality_metrics': {
#                 'ocr_confidence': primary_result['ocr_results'].get('ocr_confidence', 0.0),
#                 'tampering_confidence': primary_result['tampering_results'].get('tampering_confidence', 0.0),
#                 'page_count': len(page_results),
#                 'average_entropy': np.mean([
#                     page['ml_features']['image_stats']['entropy'] 
#                     for page in page_results
#                 ])
#             },
#             'flags': self._generate_flags(primary_result, page_results),
#             'page_summaries': [
#                 {
#                     'page': page['page_number'],
#                     'tampering_detected': page['tampering_results']['tampering_detected'],
#                     'ocr_confidence': page['ocr_results']['ocr_confidence']
#                 }
#                 for page in page_results
#             ],
#             'model_metadata': {
#                 'version': self.model_version,
#                 'config_hash': self.config_hash,
#                 'thresholds': {
#                     'reject': self.REJECT_THRESHOLD,
#                     'low_quality': self.LOW_QUALITY_THRESHOLD
#                 }
#             },
#             'recommendations': self._generate_recommendations(authenticity_score, primary_result)
#         }
        
#         return report
    
#     def _calculate_authenticity_score(self, analysis_result: Dict[str, Any]) -> float:
#         """Calculate authenticity score with configurable weights"""
#         weights = self.ocr_engine.AUTHENTICITY_WEIGHTS
        
#         ocr_quality = analysis_result['ocr_results'].get('ocr_confidence', 0.0)
#         tampering_confidence = 1.0 - analysis_result['tampering_results'].get('tampering_confidence', 0.0)
        
#         # Field completeness based on expected fields
#         extracted_fields = analysis_result['extracted_fields']
#         expected_fields = self.ocr_engine.get_expected_fields()
#         present_fields = [f for f in expected_fields if extracted_fields.get(f)]
#         field_completeness = len(present_fields) / len(expected_fields) if expected_fields else 1.0
        
#         # Calculate weighted score
#         score = (
#             weights['ocr_quality'] * ocr_quality +
#             weights['tampering'] * tampering_confidence +
#             weights['field_completeness'] * field_completeness
#         )
        
#         # Apply sigmoid for non-linearity
#         score = 1 / (1 + np.exp(-10 * (score - 0.5)))
        
#         return float(max(0.0, min(1.0, score)))
    
#     def _generate_flags(self, primary_result, all_results):
#         """Generate quality and integrity flags"""
#         flags = {
#             'tampering_detected': primary_result['tampering_results']['tampering_detected'],
#             'low_quality_scan': primary_result['ocr_results']['ocr_confidence'] < self.LOW_QUALITY_THRESHOLD,
#             'incomplete_fields': len([f for f in primary_result['extracted_fields'].values() if f]) < 2,
#             'multiple_pages': len(all_results) > 1,
#             'suspicious_patterns': self._check_suspicious_patterns(primary_result['extracted_fields']),
#             'expired_certificate': self._check_expiry(primary_result['extracted_fields'].get('expiry_date'))
#         }
#         return flags
    
#     def _check_suspicious_patterns(self, fields: Dict[str, str]) -> bool:
#         """Check for suspicious patterns in extracted data"""
#         suspicious_indicators = [
#             'test', 'example', 'sample', 'fake', 'dummy',
#             '@@', '##', '&&', 'XXXX', '00000'
#         ]
        
#         for field_value in fields.values():
#             if field_value:
#                 field_lower = field_value.lower()
#                 if any(indicator in field_lower for indicator in suspicious_indicators):
#                     return True
#         return False
    
#     def _check_expiry(self, expiry_date_str: Optional[str]) -> bool:
#         """Check if certificate is expired"""
#         if not expiry_date_str:
#             return False
        
#         try:
#             # Parse date (simplified - expand based on your date formats)
#             from dateutil import parser
#             expiry_date = parser.parse(expiry_date_str, fuzzy=True)
#             return expiry_date.date() < datetime.now().date()
#         except:
#             return False
    
#     def _generate_analysis_id(self, provider_id: str, request_id: str) -> str:
#         """Generate secure analysis ID"""
#         salt = settings.ANALYSIS_ID_SALT
#         hash_input = f"{provider_id}{request_id}{salt}{datetime.now().isoformat()}"
#         return f"anal_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"
    
#     def _generate_cache_key(self, source: Union[str, UploadFile], provider_id: str, source_type: str) -> str:
#         """Generate cache key for document"""
#         if source_type == "upload" and hasattr(source, 'filename'):
#             key_base = f"{provider_id}_{source.filename}"
#         elif source_type == "url":
#             key_base = f"{provider_id}_{source}"
#         else:
#             key_base = f"{provider_id}_{source_type}"
        
#         return f"cert_analysis:{hashlib.md5(key_base.encode()).hexdigest()}"
    
#     def _generate_recommendations(self, score: float, analysis_result: Dict) -> List[str]:
#         """Generate actionable recommendations"""
#         recommendations = []
        
#         if score < 0.3:
#             recommendations.append("Strongly recommend manual verification")
#             recommendations.append("Consider requesting original document")
        
#         if analysis_result['tampering_results']['tampering_detected']:
#             recommendations.append("Potential tampering detected - escalate to fraud team")
        
#         if analysis_result['ocr_results']['ocr_confidence'] < 0.6:
#             recommendations.append("Low OCR quality - request clearer scan")
        
#         if not recommendations:
#             recommendations.append("Proceed with standard verification")
        
#         return recommendations
    
#     async def _store_training_data(self, page_results: List[Dict], final_report: Dict):
#         """Asynchronously store data for model improvement"""
#         try:
#             training_data = {
#                 'page_results': page_results,
#                 'final_report': final_report,
#                 'timestamp': datetime.now().isoformat(),
#                 'model_version': self.model_version
#             }
            
#             # In production, send to data lake or S3
#             # Example: await data_lake_client.store(training_data)
#             logger.info(f"Training data stored for {final_report['analysis_id']}")
            
#         except Exception as e:
#             logger.error(f"Failed to store training data: {str(e)}")
#2 

# import logging
# import hashlib
# import tempfile
# import os
# from datetime import datetime
# from typing import Dict, Any, Tuple
# import aiohttp
# import aiofiles

# from .tamper_detector import PracticalTamperDetector
# from .ocr_engine import OCREngine
# from app.utils.image_processing import ImageProcessor

# logger = logging.getLogger(__name__)

# class CertificateAnalyzer:
#     def __init__(self):
#         self.tamper_detector = PracticalTamperDetector()
#         self.ocr_engine = OCREngine()
#         self.image_processor = ImageProcessor()
#         logger.info("CertificateAnalyzer initialized with all components")
    
#     async def analyze_certificate(self, document_url: str, provider_id: str, request_id: str) -> Dict[str, Any]:
#         """Main analysis pipeline"""
#         temp_path = None
#         try:
#             logger.info(f"Starting analysis pipeline for {request_id}")
            
#             # Step 1: Download document
#             temp_path = await self._download_document(document_url)
#             logger.info(f"Document downloaded to {temp_path}")
            
#             # Step 2: Preprocess document
#             processed_images = await self.image_processor.process_document(temp_path)
#             if not processed_images:
#                 raise Exception("No valid images extracted from document")
            
#             logger.info(f"Processed {len(processed_images)} images from document")
            
#             # Step 3: Analyze first page (most certificates are single page)
#             primary_image = processed_images[0]
#             analysis_results = []
            
#             for idx, image in enumerate(processed_images):
#                 logger.info(f"Analyzing page {idx + 1}")
#                 page_result = await self._analyze_single_page(image, idx)
#                 analysis_results.append(page_result)
            
#             # Step 4: Combine results from all pages
#             final_report = self._combine_analysis_results(analysis_results, provider_id, request_id)
            
#             logger.info(f"Analysis completed. Score: {final_report['authenticity_score']}")
#             return final_report
            
#         except Exception as e:
#             logger.error(f"Analysis pipeline failed: {str(e)}")
#             raise
#         finally:
#             # Cleanup temporary files
#             if temp_path and os.path.exists(temp_path):
#                 os.unlink(temp_path)
#                 logger.info(f"Cleaned up temporary file: {temp_path}")
    
#     async def _download_document(self, document_url: str) -> str:
#         """Download document from URL to temporary file"""
#         try:
#             async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
#                 async with session.get(document_url) as response:
#                     if response.status != 200:
#                         raise Exception(f"Failed to download document: HTTP {response.status}")
                    
#                     # Create temporary file
#                     fd, temp_path = tempfile.mkstemp(suffix='.doc')
#                     os.close(fd)
                    
#                     async with aiofiles.open(temp_path, 'wb') as f:
#                         async for chunk in response.content.iter_chunked(8192):
#                             await f.write(chunk)
                    
#                     return temp_path
                    
#         except Exception as e:
#             logger.error(f"Document download failed: {str(e)}")
#             raise Exception(f"Failed to download document: {str(e)}")
    
#     async def _analyze_single_page(self, image, page_num: int) -> Dict[str, Any]:
#         """Analyze a single page/image"""
#         try:
#             # Perform OCR
#             ocr_results = await self.ocr_engine.extract_text(image)
            
#             # Detect tampering
#             tampering_results = await self.tamper_detector.detect_tampering(image)
            
#             # Calculate field confidence
#             field_confidence = self._calculate_field_confidence(ocr_results['extracted_text'])
            
#             return {
#                 'page_number': page_num,
#                 'ocr_results': ocr_results,
#                 'tampering_results': tampering_results,
#                 'field_confidence': field_confidence,
#                 'extracted_fields': ocr_results['extracted_text']
#             }
            
#         except Exception as e:
#             logger.error(f"Page analysis failed for page {page_num}: {str(e)}")
#             raise
    
#     def _calculate_field_confidence(self, extracted_fields: Dict[str, str]) -> Dict[str, float]:
#         """Calculate confidence scores for extracted fields"""
#         confidence_scores = {}
        
#         for field, value in extracted_fields.items():
#             if not value or value.strip() == '':
#                 confidence_scores[field] = 0.0
#                 continue
            
#             # Calculate confidence based on field characteristics
#             base_score = 0.5
            
#             # Boost for field presence
#             if value and len(value) > 0:
#                 base_score += 0.2
            
#             # Boost for reasonable length
#             if 2 <= len(value) <= 100:
#                 base_score += 0.2
            
#             # Penalize for suspicious patterns
#             if any(char in value for char in ['@@', '##', '&&']):
#                 base_score -= 0.3
            
#             confidence_scores[field] = max(0.0, min(1.0, base_score))
        
#         return confidence_scores
    
#     def _combine_analysis_results(self, page_results: list, provider_id: str, request_id: str) -> Dict[str, Any]:
#         """Combine results from multiple pages into final report"""
#         if not page_results:
#             raise Exception("No analysis results to combine")
        
#         # Use first page as primary (most certificates are single page)
#         primary_result = page_results[0]
        
#         # Calculate overall authenticity score
#         authenticity_score = self._calculate_overall_authenticity(primary_result)
        
#         # Determine status
#         status = "Pending Admin Review" if authenticity_score >= 0.75 else "Auto-Rejected"
        
#         # Generate analysis ID
#         analysis_id = f"anal_{hashlib.md5(f'{provider_id}{request_id}'.encode()).hexdigest()[:12]}"
        
#         # Build final report
#         report = {
#             'analysis_id': analysis_id,
#             'timestamp': datetime.now().isoformat(),
#             'provider_id': provider_id,
#             'extracted_text': primary_result['extracted_fields'],
#             'authenticity_score': round(authenticity_score, 3),
#             'confidence_level': primary_result['field_confidence'],
#             'flags': {
#                 'tampering_detected': primary_result['tampering_results']['tampering_detected'],
#                 'low_quality_scan': primary_result['ocr_results']['ocr_confidence'] < 0.7,
#                 'incomplete_fields': len([f for f in primary_result['extracted_fields'].values() if f]) < 2,
#                 'multiple_pages': len(page_results) > 1
#             },
#             'detailed_scores': {
#                 'ocr_confidence': primary_result['ocr_results']['ocr_confidence'],
#                 'tampering_confidence': primary_result['tampering_results']['tampering_confidence'],
#                 'field_completeness': len([f for f in primary_result['extracted_fields'].values() if f]) / 5.0,
#                 'page_count': len(page_results)
#             },
#             'status': status,
#             'processing_time': None  # Will be set by main endpoint
#         }
        
#         return report
    
#     def _calculate_overall_authenticity(self, analysis_result: Dict[str, Any]) -> float:
#         """Calculate overall authenticity score from analysis components"""
#         ocr_quality = analysis_result['ocr_results']['ocr_confidence']
#         tampering_confidence = 1.0 - analysis_result['tampering_results']['tampering_confidence']
        
#         # Calculate field completeness
#         extracted_fields = analysis_result['extracted_fields']
#         expected_fields = ['name', 'license_number', 'expiry_date', 'issuing_authority', 'issue_date']
#         present_fields = [field for field in expected_fields if extracted_fields.get(field)]
#         field_completeness = len(present_fields) / len(expected_fields)
        
#         # Weighted authenticity score
#         authenticity_score = (
#             0.4 * ocr_quality +
#             0.4 * tampering_confidence +
#             0.2 * field_completeness
#         )
        
#         return max(0.0, min(1.0, authenticity_score))