import pytest
import asyncio
import cv2
import numpy as np
from app.analyzers.certificate_analyzer import CertificateAnalyzer
from app.analyzers.tamper_detector import PracticalTamperDetector
from app.analyzers.ocr_engine import OCREngine

@pytest.fixture
def analyzer():
    return CertificateAnalyzer()

@pytest.fixture
def sample_image():
    # Create a simple test image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Test Certificate', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

@pytest.mark.asyncio
async def test_tamper_detection(sample_image):
    detector = PracticalTamperDetector()
    results = await detector.detect_tampering(sample_image)
    
    assert 'tampering_detected' in results
    assert 'tampering_confidence' in results
    assert 0 <= results['tampering_confidence'] <= 1

@pytest.mark.asyncio
async def test_ocr_extraction(sample_image):
    engine = OCREngine()
    results = await engine.extract_text(sample_image)
    
    assert 'extracted_text' in results
    assert 'ocr_confidence' in results
    assert 0 <= results['ocr_confidence'] <= 1

@pytest.mark.asyncio
async def test_full_analysis(analyzer):
    # This would test with actual sample certificates
    # For now, just test that the analyzer initializes
    assert analyzer is not None
    assert analyzer.tamper_detector is not None
    assert analyzer.ocr_engine is not None