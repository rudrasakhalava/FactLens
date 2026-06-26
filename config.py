import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for FactLens video text extraction system.
    
    Reads from environment variables and sets defaults.
    """
    # MongoDB Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME: str = "RealityChecker"
    COLLECTION_NAME: str = "video_transcripts"
    
    # Processing Configuration
    FRAME_INTERVAL: float = float(os.getenv("FRAME_INTERVAL", 1.0))
    OCR_THRESHOLD: float = float(os.getenv("OCR_THRESHOLD", 0.4))
    
    # Whisper Configuration
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    
    # Folders
    TEMP_FOLDER: Path = Path(os.getenv("TEMP_FOLDER", "temp")).resolve()
    
    # File Formats
    SUPPORTED_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    # OCR Languages (English, Hindi, Gujarati)
    OCR_LANGUAGES: list[str] = ["en", "hi", "gu"]

    @classmethod
    def setup_temp_dir(cls) -> None:
        """Create the temporary folder if it does not exist."""
        cls.TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
