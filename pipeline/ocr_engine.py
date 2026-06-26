import logging
from typing import Dict, Any, List
import easyocr
from tqdm import tqdm
from utils.clean import remove_duplicate_ocr

logger = logging.getLogger(__name__)

class OCREngineError(Exception):
    """Exception raised when OCR processing fails."""
    pass

class OCREngine:
    """OCR engine utilizing EasyOCR to extract text from video frames."""
    
    def __init__(self, languages: List[str], threshold: float = 0.4) -> None:
        """Initialize the OCR engine with specific languages.
        
        Args:
            languages: List of language codes, e.g. ['en', 'hi', 'gu'].
            threshold: Confidence threshold below which OCR results are ignored.
        """
        self.languages = languages
        self.threshold = threshold
        self.reader = None

    def initialize_reader(self) -> None:
        """Load the EasyOCR models into memory if not already initialized.
        
        This is separated from __init__ to allow lazy loading.
        """
        if self.reader is None:
            try:
                logger.info(f"Initializing EasyOCR reader for languages: {self.languages}")
                # easyocr automatically uses GPU if available
                self.reader = easyocr.Reader(self.languages, gpu=True)
                logger.info("EasyOCR reader initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                logger.info("Retrying EasyOCR initialization with CPU-only fallback...")
                try:
                    self.reader = easyocr.Reader(self.languages, gpu=False)
                    logger.info("EasyOCR reader initialized successfully (CPU mode).")
                except Exception as ex:
                    logger.error(f"Failed CPU-only initialization of EasyOCR: {ex}")
                    raise OCREngineError(f"Could not load EasyOCR models: {ex}")

    def process_frames(self, frame_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run OCR on a list of frames, extract text, filter by confidence, and deduplicate.
        
        Args:
            frame_items: List of frame dicts with:
                         [ { 'timestamp': float, 'path': Path } ]
                         
        Returns:
            Deduplicated list of OCR detections across all frames:
            [
                {
                    "timestamp": float,
                    "text": str,
                    "confidence": float,
                    "bbox": List[List[int]]
                }
            ]
        """
        self.initialize_reader()
        if self.reader is None:
            raise RuntimeError("OCR Reader is not initialized.")
            
        raw_ocr_results: List[Dict[str, Any]] = []
        logger.info(f"Starting OCR processing for {len(frame_items)} frames...")
        
        # We use a progress bar for visual feedback
        for item in tqdm(frame_items, desc="Running OCR on frames", unit="frame"):
            timestamp = item["timestamp"]
            frame_path = item["path"]
            
            try:
                # readtext returns: [ ( [bbox], text, confidence ), ... ]
                # bbox is represented as: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                results = self.reader.readtext(str(frame_path))
                
                for bbox, text, confidence in results:
                    if confidence >= self.threshold:
                        # Convert box coordinates to standard python ints for JSON compatibility
                        converted_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
                        
                        raw_ocr_results.append({
                            "timestamp": timestamp,
                            "text": text.strip(),
                            "confidence": float(round(confidence, 4)),
                            "bbox": converted_bbox
                        })
                        
            except Exception as e:
                logger.error(f"OCR failed for frame at {timestamp}s ({frame_path.name}): {e}")
                # We do not crash the entire pipeline if a single frame fails, but we log the error.
                continue
                
        logger.info(f"Raw OCR completed. Extracted {len(raw_ocr_results)} text boxes above threshold {self.threshold}.")
        
        # Remove duplicates across nearby frames (within 3 seconds by default)
        logger.info("Applying OCR deduplication across nearby frames...")
        deduplicated_results = remove_duplicate_ocr(raw_ocr_results, similarity_threshold=0.7, time_window_seconds=3.0)
        logger.info(f"Deduplication complete. Retained {len(deduplicated_results)} unique OCR events.")
        
        return deduplicated_results
