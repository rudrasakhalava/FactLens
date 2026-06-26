import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class TranscriberError(Exception):
    """Exception raised when audio transcription fails."""
    pass

def transcribe_audio(
    audio_path: Path, 
    model_size: str = "base", 
    device: str = "cpu"
) -> Tuple[List[Dict[str, Any]], str]:
    """Transcribes an audio file using Faster Whisper.
    
    Args:
        audio_path: Path to the input WAV audio file.
        model_size: Size of the Whisper model to load (e.g. "base").
        device: Run device ("cpu" or "cuda").
        
    Returns:
        A tuple of:
          - List of segment dicts with start, end, text, and confidence.
          - Detected language code (e.g., "en").
          
    Raises:
        TranscriberError: If transcription fails or the model fails to load.
    """
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")
        
    logger.info(f"Loading Whisper model '{model_size}' on device '{device}'...")
    
    try:
        # Load Faster Whisper model
        # On CPU, float32 or int8 is typical. Let's use float32 for maximum compatibility.
        model = WhisperModel(model_size, device=device, compute_type="float32")
        logger.info("Whisper model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {e}")
        raise TranscriberError(f"Could not load Whisper model: {e}")
        
    try:
        logger.info(f"Starting audio transcription for {audio_path.name}")
        
        # Transcribe audio file. Automatic language detection is default if language is not set.
        # We set word_timestamps=True to get word-level probabilities for better confidence estimation
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True
        )
        
        detected_language = info.language
        language_probability = info.language_probability
        logger.info(f"Detected language: '{detected_language}' with probability {language_probability:.2f}")
        
        transcribed_segments = []
        for segment in segments:
            # Calculate segment confidence from word probabilities if available, otherwise fallback to logprob
            confidence = 1.0
            if segment.words:
                word_confs = [w.probability for w in segment.words if w.probability is not None]
                if word_confs:
                    confidence = sum(word_confs) / len(word_confs)
            else:
                # Fallback to exp(avg_logprob)
                confidence = math.exp(max(min(segment.avg_logprob, 0.0), -20.0))
                
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                "confidence": round(confidence, 4)
            }
            transcribed_segments.append(segment_data)
            logger.debug(f"[{segment_data['start']}s -> {segment_data['end']}s]: {segment_data['text']}")
            
        logger.info(f"Transcription completed. Extracted {len(transcribed_segments)} segments.")
        return transcribed_segments, detected_language
        
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}")
        raise TranscriberError(f"Transcription process failed: {e}")
