import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioExtractionError(Exception):
    """Exception raised when audio extraction from video fails."""
    pass

def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio track from video file as WAV format using FFmpeg.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Path to directory where audio should be saved.
        
    Returns:
        Path to the extracted WAV audio file.
        
    Raises:
        AudioExtractionError: If FFmpeg fails or is not installed.
    """
    if not shutil.which("ffmpeg"):
        logger.error("FFmpeg executable not found in system PATH. Cannot extract audio.")
        raise AudioExtractionError(
            "FFmpeg is not installed or not in system PATH. "
            "Please install FFmpeg to run audio extraction."
        )
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_path = output_dir / "audio.wav"
    
    # Overwrite if exists
    if output_audio_path.exists():
        try:
            output_audio_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete existing temp audio file: {e}")
            
    # ffmpeg command:
    # -i: input file
    # -vn: disable video recording
    # -acodec pcm_s16le: PCM signed 16-bit little-endian (standard WAV audio)
    # -ar 16000: sample rate 16000Hz (optimal for Whisper)
    # -ac 1: mono audio (optimal for Whisper)
    # -y: overwrite output files without asking
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_audio_path),
        "-y"
    ]
    
    logger.info(f"Extracting audio using command: {' '.join(cmd)}")
    
    try:
        # Run FFmpeg command silently unless there is an error
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info("Audio extracted successfully: temp/audio.wav")
        return output_audio_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg process returned error (exit code {e.returncode}): {e.stderr}")
        raise AudioExtractionError(f"FFmpeg failed to extract audio: {e.stderr}")
    except Exception as e:
        logger.error(f"Error executing FFmpeg command: {e}")
        raise AudioExtractionError(f"Unexpected error during audio extraction: {e}")
