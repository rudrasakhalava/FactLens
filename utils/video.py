import os
import logging
from pathlib import Path
from typing import Dict, Any
import cv2

logger = logging.getLogger(__name__)

class VideoMetadataError(Exception):
    """Exception raised for errors during video metadata extraction."""
    pass

def validate_video_file(file_path: str, supported_extensions: set[str]) -> Path:
    """Validate that the video file exists and has a supported extension.
    
    Args:
        file_path: Path to the video file.
        supported_extensions: Set of lowercased supported extensions, e.g. {".mp4", ".mov"}
        
    Returns:
        The validated absolute Path object.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format/extension is unsupported.
    """
    path_obj = Path(file_path).resolve()
    
    # Validate existence
    if not path_obj.exists() or not path_obj.is_file():
        logger.error(f"Video file not found: {file_path}")
        raise FileNotFoundError(f"Video file does not exist: {file_path}")
        
    # Validate extension
    ext = path_obj.suffix.lower()
    if ext not in supported_extensions:
        logger.error(f"Unsupported video extension '{ext}' for file: {file_path}")
        raise ValueError(
            f"Unsupported video format '{ext}'. Supported formats: {', '.join(supported_extensions)}"
        )
        
    return path_obj

def get_video_metadata(file_path: Path) -> Dict[str, Any]:
    """Retrieve video metadata including duration, fps, frame count, resolution, codec, and size.
    
    Args:
        file_path: Path to the validated video file.
        
    Returns:
        Dict containing metadata fields.
        
    Raises:
        VideoMetadataError: If metadata cannot be retrieved or video is corrupted.
    """
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        logger.error(f"OpenCV could not open video file: {file_path}")
        raise VideoMetadataError(f"Could not open video file '{file_path}'. It might be corrupted.")
        
    try:
        # Get frame properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Decode codec (fourcc)
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ""
        if fourcc_val != 0:
            codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)]).strip()
            
        # Compute duration
        duration = 0.0
        if fps > 0:
            duration = frame_count / fps
            
        # File size in bytes
        file_size_bytes = os.path.getsize(file_path)
        
        # Verify if video parameters make sense (simple corruption detection)
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            logger.error(f"Invalid video structure: fps={fps}, frames={frame_count}, dimensions={width}x{height}")
            raise VideoMetadataError("Video structure is invalid or corrupted (missing frame/fps metadata).")
            
        metadata = {
            "filename": file_path.name,
            "path": str(file_path),
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "codec": codec if codec else "unknown",
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_bytes / (1024 * 1024), 2),
        }
        
        logger.info(f"Successfully extracted video metadata: {metadata['filename']} ({metadata['resolution']}, {metadata['duration']}s)")
        return metadata
        
    except Exception as e:
        logger.error(f"Error reading metadata from video {file_path}: {e}")
        raise VideoMetadataError(f"Failed to read video metadata: {e}")
    finally:
        cap.release()
