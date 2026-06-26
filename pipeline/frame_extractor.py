import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
import cv2

logger = logging.getLogger(__name__)

class FrameExtractionError(Exception):
    """Exception raised when video frame extraction fails."""
    pass

def extract_frames(
    video_path: Path, 
    output_dir: Path, 
    interval_seconds: float = 1.0
) -> List[Dict[str, Any]]:
    """Extract frames from video at regular intervals and save them temporarily.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory where frames will be saved.
        interval_seconds: Time interval between extracted frames in seconds.
        
    Returns:
        List of dicts representing extracted frames:
        [
            {
                "timestamp": float,
                "path": Path,
                "frame_number": int
            }
        ]
        
    Raises:
        FrameExtractionError: If OpenCV fails to read the video or write frames.
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found for frame extraction: {video_path}")
        
    frames_dir = output_dir / "frames"
    
    # Re-create frames directory to ensure it is clean
    if frames_dir.exists():
        try:
            shutil.rmtree(frames_dir)
        except Exception as e:
            logger.warning(f"Could not clean existing frames directory: {e}")
            
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        raise FrameExtractionError(f"OpenCV failed to open video file '{video_path}'")
        
    extracted_frames: List[Dict[str, Any]] = []
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            raise FrameExtractionError("Invalid video properties (FPS or frame count is zero).")
            
        duration = total_frames / fps
        logger.info(f"Extracting frames from video: {video_path.name} (FPS: {fps:.2f}, Duration: {duration:.2f}s)")
        logger.info(f"Target interval: {interval_seconds}s")
        
        # Calculate frame step based on interval and fps
        frame_step = max(1, int(round(fps * interval_seconds)))
        
        frame_idx = 0
        extracted_count = 0
        
        while cap.isOpened():
            # Set video position to specific frame to optimize speed instead of reading every single frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            timestamp = round(frame_idx / fps, 2)
            frame_filename = f"frame_{extracted_count:05d}_{timestamp:.2f}s.jpg"
            frame_path = frames_dir / frame_filename
            
            # Save frame to disk
            success = cv2.imwrite(str(frame_path), frame)
            if not success:
                logger.error(f"Failed to write frame at {timestamp}s to {frame_path}")
                raise FrameExtractionError(f"Could not write frame file: {frame_path}")
                
            extracted_frames.append({
                "timestamp": timestamp,
                "path": frame_path,
                "frame_number": frame_idx
            })
            
            extracted_count += 1
            frame_idx += frame_step
            
            if frame_idx >= total_frames:
                break
                
        logger.info(f"Frame extraction completed. Extracted {len(extracted_frames)} frames.")
        return extracted_frames
        
    except Exception as e:
        logger.error(f"Error occurred during frame extraction: {e}")
        # Clean up directory on failure
        cleanup_frames_dir(frames_dir)
        raise FrameExtractionError(f"Frame extraction process failed: {e}")
    finally:
        cap.release()

def cleanup_frames_dir(frames_dir: Path) -> None:
    """Delete the directory containing extracted frames and all its contents.
    
    Args:
        frames_dir: Path to the frames folder.
    """
    if frames_dir.exists() and frames_dir.is_dir():
        try:
            shutil.rmtree(frames_dir)
            logger.info("Temporary frames directory cleaned up successfully.")
        except Exception as e:
            logger.warning(f"Failed to delete temporary frames directory '{frames_dir}': {e}")
