import time
import logging
from typing import Any
from google.genai import types

logger = logging.getLogger(__name__)

def generate_content_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
    initial_delay: float = 15.0
) -> Any:
    """Helper function to execute Gemini content generation with backoff on 429 rate limits.
    
    Args:
        client: The google-genai Client instance.
        model: Model identifier.
        contents: Input contents prompt.
        config: GenerateContentConfig instance.
        max_retries: Number of retries.
        initial_delay: Initial sleep delay in seconds.
        
    Returns:
        The response object.
        
    Raises:
        Exception: If retries are exhausted or other non-quota errors occur.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            err_str = str(e)
            is_transient = (
                "429" in err_str or 
                "RESOURCE_EXHAUSTED" in err_str or 
                "503" in err_str or
                "UNAVAILABLE" in err_str or
                "500" in err_str or
                "quota" in err_str.lower() or 
                "rate limit" in err_str.lower() or
                "high demand" in err_str.lower()
            )
            
            is_daily_limit = (
                "perday" in err_str.lower() or 
                "per day" in err_str.lower() or 
                "daily" in err_str.lower()
            )
            
            if is_transient and not is_daily_limit and attempt < max_retries:
                logger.warning(
                    f"Gemini API transient failure (429/503/500) on attempt {attempt}/{max_retries}. "
                    f"Sleeping for {delay} seconds before retrying..."
                )
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"Gemini API generation failed after {attempt} attempts: {e}")
                raise e
    
    raise RuntimeError(f"Failed to generate content after {max_retries} attempts due to rate limits.")
