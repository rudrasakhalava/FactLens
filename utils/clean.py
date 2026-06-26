import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List

def normalize_spaces(text: str) -> str:
    """Normalize spaces in a string by collapsing multiple spaces and newlines.
    
    Args:
        text: Input string.
        
    Returns:
        Cleaned string with single spaces and normalized whitespace.
    """
    if not text:
        return ""
    # Collapse multiple whitespaces/newlines into a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def fix_unicode(text: str) -> str:
    """Fix Unicode normalization issues.
    
    Args:
        text: Input string.
        
    Returns:
        Unicode NFKC normalized string.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)

def clean_text(text: str) -> str:
    """Combine space normalization, Unicode fixing, and general cleaning.
    
    Args:
        text: Input string.
        
    Returns:
        Cleaned string, or empty string.
    """
    text = fix_unicode(text)
    text = normalize_spaces(text)
    return text

def is_similar(str1: str, str2: str, threshold: float = 0.7) -> bool:
    """Check if two strings are similar using SequenceMatcher.
    
    Args:
        str1: First string.
        str2: Second string.
        threshold: Similarity threshold (0.0 to 1.0).
        
    Returns:
        True if similarity is greater than or equal to threshold, False otherwise.
    """
    # Quick length check to optimize performance
    len1, len2 = len(str1), len(str2)
    if len1 == 0 or len2 == 0:
        return False
    if min(len1, len2) / max(len1, len2) < threshold - 0.2:
        return False
    return SequenceMatcher(None, str1, str2).ratio() >= threshold

def remove_duplicate_ocr(ocr_items: List[Dict[str, Any]], similarity_threshold: float = 0.7, time_window_seconds: float = 3.0) -> List[Dict[str, Any]]:
    """Deduplicates OCR results that appear in nearby frames and contain highly similar text.
    
    Args:
        ocr_items: List of dicts, where each dict has:
                   { 'timestamp': float, 'text': str, 'confidence': float, 'bbox': list }
        similarity_threshold: Similarity ratio above which texts are considered duplicate.
        time_window_seconds: Time window in seconds within which duplicates are checked.
        
    Returns:
        Deduplicated list of OCR items.
    """
    if not ocr_items:
        return []
        
    # Sort by timestamp
    sorted_items = sorted(ocr_items, key=lambda x: x["timestamp"])
    deduplicated: List[Dict[str, Any]] = []
    
    for item in sorted_items:
        text = clean_text(item.get("text", ""))
        if not text:
            continue
            
        # Update the item with cleaned text
        item["text"] = text
        
        # Check against already added items in the recent window
        is_dup = False
        for prev in reversed(deduplicated):
            # Since items are sorted, we can stop if we are outside the window
            if item["timestamp"] - prev["timestamp"] > time_window_seconds:
                break
                
            if is_similar(text, prev["text"], similarity_threshold):
                is_dup = True
                # If duplicate, keep the one with higher confidence
                if item.get("confidence", 0) > prev.get("confidence", 0):
                    prev["text"] = item["text"]
                    prev["confidence"] = item["confidence"]
                    prev["bbox"] = item.get("bbox")
                    prev["timestamp"] = item["timestamp"]  # Update to latest occurrence
                break
                
        if not is_dup:
            deduplicated.append(item)
            
    return deduplicated

def remove_repeated_speech(speech_segments: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Remove consecutive or highly similar repeated speech segments.
    
    Args:
        speech_segments: List of speech dicts with { 'start': float, 'end': float, 'text': str }
        similarity_threshold: Similarity ratio above which texts are considered duplicate.
        
    Returns:
        Deduplicated list of speech segments.
    """
    if not speech_segments:
        return []
        
    sorted_segments = sorted(speech_segments, key=lambda x: x["start"])
    cleaned_segments: List[Dict[str, Any]] = []
    
    for segment in sorted_segments:
        text = clean_text(segment.get("text", ""))
        if not text:
            continue
            
        segment["text"] = text
        
        if not cleaned_segments:
            cleaned_segments.append(segment)
            continue
            
        prev_segment = cleaned_segments[-1]
        
        # If speech is consecutive and highly similar, merge or discard the duplicate
        if is_similar(text, prev_segment["text"], similarity_threshold):
            # Extend the duration of the previous speech segment
            prev_segment["end"] = max(prev_segment["end"], segment["end"])
            # Keep the longer text or simply merge
            if len(text) > len(prev_segment["text"]):
                prev_segment["text"] = text
        else:
            cleaned_segments.append(segment)
            
    return cleaned_segments
