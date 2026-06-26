import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the project root directory to python path if run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from database.mongo_client import MongoDatabase
from utils.video import validate_video_file, get_video_metadata, VideoMetadataError
from utils.clean import remove_repeated_speech
from pipeline.audio_extractor import extract_audio, AudioExtractionError
from pipeline.transcriber import transcribe_audio, TranscriberError
from pipeline.frame_extractor import extract_frames, FrameExtractionError, cleanup_frames_dir
from pipeline.ocr_engine import OCREngine, OCREngineError
from pipeline.merger import merge_speech_and_ocr, compile_complete_text

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("factlens_processing.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("FactLensOrchestrator")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="FactLens: Advanced Video Text & Speech Metadata Extractor"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (supports mp4, mov, avi, mkv, webm)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=Config.FRAME_INTERVAL,
        help=f"Frame extraction interval in seconds (default: {Config.FRAME_INTERVAL})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.OCR_THRESHOLD,
        help=f"EasyOCR confidence threshold (default: {Config.OCR_THRESHOLD})"
    )
    return parser.parse_args()

def process_video_pipeline(video_path_str: str, interval: float, threshold: float) -> Dict[str, Any]:
    """Execute the complete FactLens video processing pipeline.
    
    Args:
        video_path_str: Path to the input video.
        interval: Frame extraction interval in seconds.
        threshold: OCR confidence threshold.
        
    Returns:
        Dict: Complete document that was generated and prepared for database insertion.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting FactLens pipeline for: {video_path_str}")
    logger.info("=" * 60)

    # 0. Setup directories
    Config.setup_temp_dir()
    
    # 1. Video Loading and Validation (Feature 1)
    try:
        video_path = validate_video_file(video_path_str, Config.SUPPORTED_EXTENSIONS)
        metadata = get_video_metadata(video_path)
    except FileNotFoundError as e:
        logger.error(f"Validation failed - File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation failed - Unsupported format: {e}")
        sys.exit(1)
    except VideoMetadataError as e:
        logger.error(f"Validation failed - Video corrupted or unreadable: {e}")
        sys.exit(1)
        
    # 2. Audio Extraction (Feature 2)
    temp_audio_path = None
    try:
        temp_audio_path = extract_audio(video_path, Config.TEMP_FOLDER)
    except AudioExtractionError as e:
        logger.error(f"Pipeline stalled during Audio Extraction: {e}")
        # Note: We continue without audio if audio extraction fails, or we can choose to fail.
        # Clean architecture dictates we raise or fail gracefully depending on criticality.
        # Since transcription relies on audio, this is a blocker.
        sys.exit(1)

    # 3. Speech Recognition (Feature 3)
    speech_segments: List[Dict[str, Any]] = []
    detected_language = "unknown"
    if temp_audio_path and temp_audio_path.exists():
        try:
            raw_speech, detected_language = transcribe_audio(
                audio_path=temp_audio_path,
                model_size=Config.WHISPER_MODEL,
                device="cpu"
            )
            # Text Cleaning: Remove repeated speech segments (Feature 13)
            speech_segments = remove_repeated_speech(raw_speech)
        except TranscriberError as e:
            logger.error(f"Speech recognition step failed: {e}")
            # Whisper failure shouldn't necessarily block OCR if we want partial results,
            # but per instructions, we handle it as a major failure.
            speech_segments = []

    # 4. Frame Extraction (Feature 4)
    frame_items: List[Dict[str, Any]] = []
    try:
        frame_items = extract_frames(
            video_path=video_path,
            output_dir=Config.TEMP_FOLDER,
            interval_seconds=interval
        )
    except FrameExtractionError as e:
        logger.error(f"Frame extraction failed: {e}")
        frame_items = []

    # 5. Visual Text OCR (Feature 5)
    ocr_results: List[Dict[str, Any]] = []
    if frame_items:
        try:
            ocr_engine = OCREngine(languages=Config.OCR_LANGUAGES, threshold=threshold)
            ocr_results = ocr_engine.process_frames(frame_items)
        except OCREngineError as e:
            logger.error(f"OCR step failed: {e}")
            ocr_results = []
        finally:
            # Delete temporary frames folder after OCR processing (Feature 4)
            frames_dir = Config.TEMP_FOLDER / "frames"
            cleanup_frames_dir(frames_dir)

    # Clean up extracted audio file
    if temp_audio_path and temp_audio_path.exists():
        try:
            temp_audio_path.unlink()
            logger.info("Temporary audio WAV file deleted.")
        except Exception as e:
            logger.warning(f"Could not delete temporary audio file: {e}")

    # 6. Merge Speech and OCR Results (Feature 6)
    merged_transcript = merge_speech_and_ocr(speech_segments, ocr_results)
    
    # 7. Compile Final Unified Text
    complete_text = compile_complete_text(merged_transcript)
    
    processing_time = round(time.time() - start_time, 2)
    
    # Construct final document structure (Feature 7)
    document = {
        "filename": metadata["filename"],
        "path": metadata["path"],
        "duration": metadata["duration"],
        "fps": metadata["fps"],
        "resolution": metadata["resolution"],
        "language": detected_language,
        "speech": speech_segments,
        "ocr": ocr_results,
        "merged_transcript": merged_transcript,
        "complete_text": complete_text,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "processing_time": f"{processing_time}s",
        "model": f"faster-whisper-{Config.WHISPER_MODEL}"
    }
    
    return document

def main() -> None:
    """Main execution orchestrator."""
    args = parse_arguments()
    
    # Process the video pipeline
    document = process_video_pipeline(
        video_path_str=args.video_path,
        interval=args.interval,
        threshold=args.threshold
    )
    
    # 8. Persist to MongoDB (Feature 7 & 14)
    db_id = "Not Inserted"
    db = MongoDatabase(uri=Config.MONGO_URI, db_name=Config.DB_NAME, collection_name=Config.COLLECTION_NAME)
    try:
        db.connect()
        db_id = db.insert_video(document)
    except ConnectionError as e:
        logger.error(f"Database insertion skipped: MongoDB disconnected or unavailable. Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing to MongoDB: {e}")
    finally:
        db.close()

    # 9. Output Statistics display (Feature 18)
    print("\n" + "=" * 50)
    print("                 FACTLENS SUMMARY                ")
    print("=" * 50)
    print(f"Video File:       {document['filename']}")
    print(f"Resolution:       {document['resolution']}")
    print(f"FPS:              {document['fps']}")
    print(f"Duration:         {document['duration']} seconds")
    print(f"Language:         {document['language']}")
    print(f"Speech Segments:  {len(document['speech'])}")
    print(f"OCR Detections:   {len(document['ocr'])}")
    print(f"Processing Time:  {document['processing_time']}")
    print(f"MongoDB ObjectId: {db_id}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
