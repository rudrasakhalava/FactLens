def format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted string, e.g. "01:23" or "12:34:56".
    """
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string (HH:MM:SS or MM:SS) back to seconds.
    
    Args:
        timestamp: Time string.
        
    Returns:
        Equivalent time in float seconds.
    """
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        return float(parts[0])
