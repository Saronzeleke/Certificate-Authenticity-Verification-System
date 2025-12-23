import logging
import cv2
import numpy as np
from pdf2image import convert_from_path
import tempfile
import os
from typing import List

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp']
        logger.info("ImageProcessor initialized")
    
    async def process_document(self, document_path: str) -> List[np.ndarray]:
        """Process document and extract images for analysis"""
        try:
            file_ext = os.path.splitext(document_path)[1].lower()
            
            if file_ext == '.pdf':
                return await self._process_pdf(document_path)
            else:
                return await self._process_image(document_path)
                
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def _process_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF to images"""
        try:
            # Convert PDF to images (first 3 pages max for performance)
            images = convert_from_path(pdf_path, first_page=1, last_page=3, dpi=200)
            
            processed_images = []
            for img in images:
                # Convert PIL to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                processed_images.append(cv_img)
            
            logger.info(f"Converted PDF to {len(processed_images)} images")
            return processed_images
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    async def _process_image(self, image_path: str) -> List[np.ndarray]:
        """Process single image file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load image")
            
            # Basic validation
            if image.size == 0:
                raise Exception("Empty image")
            
            # Resize if too large (max 2000px on longer side)
            h, w = image.shape[:2]
            if max(h, w) > 2000:
                scale = 2000 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return [image]
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise Exception(f"Failed to process image: {str(e)}")
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image for better OCR"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Threshold and find skew angle
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 100:  # Not enough points for reliable deskewing
                return image
            
            # Calculate skew angle
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image  # Return original if deskewing fails