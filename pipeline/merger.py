import logging
from typing import Dict, Any, List
from utils.time_format import format_seconds_to_timestamp

logger = logging.getLogger(__name__)

def merge_speech_and_ocr(
    speech_segments: List[Dict[str, Any]], 
    ocr_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge speech segments and OCR results chronologically.
    
    If OCR text falls within a speech segment's start and end timestamps,
    it is merged into the same entry under "visual".
    Otherwise, OCR text is created as an independent entry with empty speech.
    
    Args:
        speech_segments: List of speech segments with start, end, text.
        ocr_results: List of OCR items with timestamp, text, confidence, bbox.
        
    Returns:
        Sorted list of merged transcript entries:
        [
            {
                "timestamp": float,
                "timestamp_formatted": str,
                "speech": str,
                "visual": str
            }
        ]
    """
    logger.info("Merging speech transcription and OCR visual text...")
    
    # Sort inputs
    sorted_speech = sorted(speech_segments, key=lambda x: x["start"])
    sorted_ocr = sorted(ocr_results, key=lambda x: x["timestamp"])
    
    merged_entries: List[Dict[str, Any]] = []
    consumed_ocr_indices = set()
    
    # Process speech segments first
    for speech in sorted_speech:
        start_time = speech["start"]
        end_time = speech["end"]
        speech_text = speech["text"]
        
        # Find all OCR texts that fall within this speech segment's timeframe
        associated_ocr_texts = []
        for idx, ocr in enumerate(sorted_ocr):
            # Check if OCR timestamp falls inside [start_time, end_time]
            # Add a minor buffer of 0.5 seconds on either side to handle alignment overlap
            if (start_time - 0.5) <= ocr["timestamp"] <= (end_time + 0.5):
                associated_ocr_texts.append(ocr["text"])
                consumed_ocr_indices.add(idx)
                
        visual_text = " | ".join(associated_ocr_texts) if associated_ocr_texts else ""
        
        merged_entries.append({
            "timestamp": start_time,
            "timestamp_formatted": format_seconds_to_timestamp(start_time),
            "speech": speech_text,
            "visual": visual_text
        })
        
    # Process remaining OCR elements that were not associated with any speech segment
    for idx, ocr in enumerate(sorted_ocr):
        if idx not in consumed_ocr_indices:
            merged_entries.append({
                "timestamp": ocr["timestamp"],
                "timestamp_formatted": format_seconds_to_timestamp(ocr["timestamp"]),
                "speech": "",
                "visual": ocr["text"]
            })
            
    # Sort all merged entries chronologically by their timestamp
    merged_entries.sort(key=lambda x: x["timestamp"])
    logger.info(f"Merging complete. Generated {len(merged_entries)} chronological timeline entries.")
    
    return merged_entries

def compile_complete_text(merged_transcript: List[Dict[str, Any]]) -> str:
    """Compile a single unified text transcript representing the complete video timeline.
    
    Args:
        merged_transcript: The output from merge_speech_and_ocr.
        
    Returns:
        Formatted multi-line text document.
    """
    lines = []
    for entry in merged_transcript:
        timestamp_str = entry["timestamp_formatted"]
        speech = entry["speech"]
        visual = entry["visual"]
        
        parts = []
        if speech:
            parts.append(f'Speech: "{speech}"')
        if visual:
            parts.append(f'Visual Text: [{visual}]')
            
        if parts:
            lines.append(f"[{timestamp_str}] " + " | ".join(parts))
            
    return "\n".join(lines)
