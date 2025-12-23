import logging
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Dict, Any, Tuple, Optional
import asyncio
from datetime import datetime
import os
import tempfile
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ProductionTamperDetector:
    """
    Production-grade tamper detector optimized for certificates/documents.
    Focuses on actual forgery patterns, not compression artifacts.
    """
    
    def __init__(self, config=None):
        self.model_version = "2024.1.0-cert-prod"
        self.config = config or {}
        
        # ADJUSTED THRESHOLDS - Less sensitive for certificates
        self.TAMPER_THRESHOLD = 0.75  # Increased from 0.6
        self.ELA_THRESHOLD = 35.0  # Increased from 25.0
        self.NOISE_CONSISTENCY_THRESHOLD = 25.0  # Increased from 15.0
        
        # Revised weights - focus on actual forgery patterns
        self.weights = {
            'ela': 0.25,  # Reduced weight
            'noise': 0.20,  # Reduced weight
            'consistency': 0.25,
            'copy_move': 0.30,  # NEW: Copy-move detection
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ProductionTamperDetector v{self.model_version} initialized (CPU/GPU: {self.device})")
    
    async def detect_tampering(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main detection pipeline optimized for certificates
        """
        try:
            if image is None or image.size == 0:
                return self._get_fallback_response()
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run detectors in parallel
            ela_task = asyncio.to_thread(self._extract_ela_score, processed_image)
            noise_task = asyncio.to_thread(self._extract_noise_analysis, processed_image)
            consistency_task = asyncio.to_thread(self._check_certificate_consistency, processed_image)
            copy_move_task = asyncio.to_thread(self._detect_copy_move, processed_image)
            
            ela_score, noise_score, consistency_score, copy_move_score = await asyncio.gather(
                ela_task, noise_task, consistency_task, copy_move_task
            )
            
            # Calculate final score with certificate-specific logic
            final_score = self._calculate_certificate_score(
                ela_score, noise_score, consistency_score, copy_move_score
            )
            
            # Determine if tampered (with higher confidence requirement)
            is_tampered = final_score > self.TAMPER_THRESHOLD
            
            # Generate detailed flags
            flags = self._generate_certificate_flags(
                ela_score, noise_score, consistency_score, copy_move_score
            )
            
            return {
                "tampering_detected": bool(is_tampered),
                "tampering_confidence": float(final_score),
                "detailed_scores": {
                    "ela_score": float(ela_score),
                    "noise_anomaly_score": float(noise_score),
                    "statistical_consistency": float(consistency_score),
                    "copy_move_detection": float(copy_move_score)
                },
                "flags": flags,
                "thresholds": {
                    "tamper": self.TAMPER_THRESHOLD,
                    "warning": 0.65
                },
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "analysis_notes": "Optimized for certificate/document analysis"
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=False)
            return self._get_fallback_response()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for certificate analysis"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent processing (preserve aspect ratio)
            h, w = image.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception:
            return image if hasattr(image, 'shape') else np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _extract_ela_score(self, image: np.ndarray) -> float:
        """
        Enhanced ELA specifically for certificates - less sensitive to JPEG artifacts
        Focuses on unnatural edges that indicate copy-paste
        """
        try:
            # Use PNG for better quality preservation
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            
            # Save as PNG (lossless)
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Save as JPEG with moderate quality
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            compressed = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            os.unlink(path)
            
            if original is None or compressed is None:
                return 0.0
            
            # Calculate difference
            diff = cv2.absdiff(original, compressed)
            
            # Focus on high-frequency areas (edges, text) - not uniform areas
            edges = cv2.Canny(original, 50, 150)
            edge_mask = (edges > 0).astype(np.uint8)
            
            if np.sum(edge_mask) > 100:  # If we have enough edges
                edge_diff = diff[edge_mask == 1]
                if len(edge_diff) > 0:
                    edge_score = np.mean(edge_diff) / 255.0
                else:
                    edge_score = 0.0
            else:
                edge_score = 0.0
            
            # Overall difference score (lower weight)
            overall_score = np.mean(diff) / 255.0
            
            # Combine with emphasis on edge anomalies
            combined_score = 0.7 * edge_score + 0.3 * overall_score
            
            # Sigmoid normalization
            normalized = 1 / (1 + np.exp(-12 * (combined_score - 0.1)))
            
            return float(min(normalized, 1.0))
            
        except Exception as e:
            logger.debug(f"ELA extraction failed: {e}")
            return 0.3  # Return neutral score instead of 0.5
    
    def _extract_noise_analysis(self, image: np.ndarray) -> float:
        """
        Noise analysis optimized for certificates
        Certificates can have varying noise levels, so be more tolerant
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate noise using wavelet-like decomposition
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            noise = cv2.absdiff(gray, denoised)
            
            # Analyze noise distribution
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Certificates typically have moderate, consistent noise
            # Very low noise (perfect scans) or very high noise (scrambled) are suspicious
            if noise_mean < 2.0:  # Too perfect
                score = 0.6
            elif noise_mean > 20.0:  # Too noisy
                score = 0.7
            elif noise_std > 15.0:  # Inconsistent noise
                score = 0.65
            else:
                score = 0.2  # Normal certificate noise
            
            return float(min(score, 1.0))
            
        except Exception:
            return 0.3
    
    def _check_certificate_consistency(self, image: np.ndarray) -> float:
        """
        Check if image has characteristics of a real certificate
        Real certificates have: text, signatures, seals, structured layout
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            score = 0.5  # Start neutral
            
            # 1. Text detection (certificates should have text)
            # Use adaptive threshold to find text regions
            binary = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            
            # Count reasonable text-like components
            text_like = 0
            for i in range(1, num_labels):
                if 10 < stats[i, cv2.CC_STAT_AREA] < 10000:  # Reasonable text size
                    text_like += 1
            
            text_density = text_like / (h * w / 10000)  # Components per 10000 pixels
            
            if text_density > 5:  # Good amount of text
                score += 0.2
            elif text_density < 1:  # Very little text (suspicious)
                score -= 0.2
            
            # 2. Aspect ratio check (common certificate sizes)
            aspect_ratio = w / h
            common_ratios = [1.2941, 1.4142, 1.0, 1.5, 0.7071]  # A4, A5, square, etc
            
            closest_ratio = min(common_ratios, key=lambda x: abs(x - aspect_ratio))
            if abs(aspect_ratio - closest_ratio) < 0.15:
                score += 0.1
            else:
                score -= 0.1
            
            # 3. Edge consistency (certificates have structured edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if 0.05 <= edge_density <= 0.3:  # Normal range for certificates
                score += 0.1
            else:
                score -= 0.1
            
            # 4. Color analysis (certificates often have some color elements)
            if len(image.shape) == 3:
                color_std = np.std(image, axis=(0, 1))
                color_variance = np.mean(color_std)
                if color_variance > 10:  # Some color variation
                    score += 0.1
            
            return float(max(0.0, min(1.0, score)))
            
        except Exception:
            return 0.5
    
    def _detect_copy_move(self, image: np.ndarray) -> float:
        """
        Detect copy-move forgery specific to certificates
        Common in certificate forgeries: copied signatures, seals, text
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use SIFT or ORB to detect duplicate features
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 10:
                return 0.1  # Not enough features to analyze
            
            # Use FLANN matcher
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Match descriptors with themselves
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            
            # Find good matches (excluding self-matches)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                        good_matches.append(m)
            
            # Analyze spatial distribution of matches
            if len(good_matches) > 5:
                # Calculate distances between matched points
                distances = []
                for match in good_matches:
                    pt1 = keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    distances.append(distance)
                
                avg_distance = np.mean(distances) if distances else 0
                
                # If many close matches with similar distances, could be copy-move
                if avg_distance > 20 and len(good_matches) > len(keypoints) * 0.05:
                    return min(0.8, len(good_matches) / 100.0)
            
            return 0.1  # Low probability of copy-move
            
        except Exception:
            return 0.1
    
    def _calculate_certificate_score(self, ela: float, noise: float, 
                                   consistency: float, copy_move: float) -> float:
        """
        Calculate tampering score optimized for certificates
        Higher thresholds, more conservative detection
        """
        # Weight the scores
        weighted = (
            self.weights['ela'] * ela +
            self.weights['noise'] * noise +
            self.weights['consistency'] * (1 - consistency) +  # Inconsistency = suspicious
            self.weights['copy_move'] * copy_move
        )
        
        # Apply conservative sigmoid
        score = 1 / (1 + np.exp(-10 * (weighted - 0.6)))  # Shifted threshold
        
        return float(max(0.0, min(1.0, score)))
    
    def _generate_certificate_flags(self, ela: float, noise: float, 
                                  consistency: float, copy_move: float) -> Dict[str, bool]:
        """Generate specific flags for certificate tampering"""
        return {
            "high_ela_anomaly": ela > 0.7,
            "suspicious_noise_pattern": noise > 0.6,
            "layout_inconsistency": consistency < 0.4,
            "possible_copy_move": copy_move > 0.5,
            "low_text_density": consistency < 0.3,  # From certificate consistency check
            "unnatural_edges": ela > 0.6 and noise > 0.4
        }
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Return safe fallback response"""
        return {
            "tampering_detected": False,
            "tampering_confidence": 0.0,
            "detailed_scores": {
                "ela_score": 0.0,
                "noise_anomaly_score": 0.0,
                "statistical_consistency": 0.5,
                "copy_move_detection": 0.0
            },
            "flags": {},
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "analysis_notes": "Fallback analysis - insufficient data"
        }


class FastTamperDetector:
    """
    Fast, lightweight detector for production use when speed is critical
    Uses simplified heuristics optimized for certificates
    """
    
    def __init__(self):
        self.model_version = "2024.1.0-fast"
    
    async def detect_tampering(self, image: np.ndarray) -> Dict[str, Any]:
        """Fast tampering detection for certificates"""
        try:
            if image is None or image.size == 0:
                return self._get_fallback_response()
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Simple ELA
            ela_score = self._fast_ela(gray)
            
            # Edge consistency check
            edge_score = self._edge_consistency(gray)
            
            # Text density check
            text_score = self._text_density_check(gray)
            
            # Combine scores (conservative)
            tampering_confidence = 0.4 * ela_score + 0.3 * edge_score + 0.3 * text_score
            
            # Only flag if strong evidence
            is_tampered = tampering_confidence > 0.75
            
            return {
                'tampering_detected': bool(is_tampered),
                'tampering_confidence': float(tampering_confidence),
                'detailed_scores': {
                    'ela_score': float(ela_score),
                    'edge_consistency': float(edge_score),
                    'text_density': float(text_score)
                },
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fast detection failed: {str(e)}")
            return self._get_fallback_response()
    
    def _fast_ela(self, image: np.ndarray) -> float:
        """Fast ELA for certificates"""
        try:
            # Simple recompression test
            _, buffer1 = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, buffer2 = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            img1 = cv2.imdecode(buffer1, 0)
            img2 = cv2.imdecode(buffer2, 0)
            
            if img1 is None or img2 is None:
                return 0.3
            
            diff = np.abs(img1.astype(float) - img2.astype(float))
            score = np.mean(diff) / 100.0
            
            # Conservative scoring
            return float(min(score, 1.0))
            
        except:
            return 0.3
    
    def _edge_consistency(self, image: np.ndarray) -> float:
        """Check edge consistency"""
        try:
            edges = cv2.Canny(image, 50, 150)
            
            # Divide into quadrants
            h, w = image.shape
            quadrants = [
                edges[0:h//2, 0:w//2],
                edges[0:h//2, w//2:w],
                edges[h//2:h, 0:w//2],
                edges[h//2:h, w//2:w]
            ]
            
            densities = [np.mean(q > 0) for q in quadrants]
            
            # Check consistency across quadrants
            if len(densities) >= 2:
                std_density = np.std(densities)
                # High std = inconsistent edges = suspicious
                return float(min(std_density * 10, 1.0))
            
            return 0.3
            
        except:
            return 0.3
    
    def _text_density_check(self, image: np.ndarray) -> float:
        """Check if image has reasonable text density"""
        try:
            # Simple thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Count connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            
            if num_labels < 2:
                return 0.7  # No text = suspicious
            
            # Count text-like components
            text_count = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 50 < area < 5000:  # Reasonable text size range
                    text_count += 1
            
            # Normalize
            h, w = image.shape
            density = text_count / (h * w / 10000)  # Components per 10000 pixels
            
            if density < 2:  # Very low text
                return 0.6
            elif density > 50:  # Unusually high
                return 0.5
            else:
                return 0.2  # Normal
            
        except:
            return 0.3
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        return {
            'tampering_detected': False,
            'tampering_confidence': 0.0,
            'detailed_scores': {},
            'model_version': self.model_version
        }


# Factory function to get appropriate detector
def get_tamper_detector(mode: str = "production", config: dict = None):
    """
    Factory function to get appropriate tamper detector
    
    Args:
        mode: "production" (default), "fast", or "legacy"
        config: Optional configuration dictionary
    """
    if mode.lower() == "fast":
        return FastTamperDetector()
    elif mode.lower() == "production":
        return ProductionTamperDetector(config)
    else:
        # Fallback to simple detector
        return FastTamperDetector()