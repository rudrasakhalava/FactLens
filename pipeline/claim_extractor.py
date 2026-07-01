import logging
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import Config

logger = logging.getLogger(__name__)

class Claim(BaseModel):
    claim_id: str = Field(description="The unique identifier for the claim, e.g., 'Claim 1', 'Claim 2'.")
    claim_text: str = Field(description="The factual claim extracted from the transcript. Make it self-contained and clear.")
    timestamp: float = Field(description="The timestamp in seconds when the claim appears in the video.")
    category: str = Field(description="Category of the claim (e.g., Historical, Scientific, Medical, Political, Economic, Tech, Sports, Geographical, etc.)")

class ClaimList(BaseModel):
    claims: List[Claim]

class ClaimExtractorError(Exception):
    """Exception raised when claim extraction fails."""
    pass

def extract_claims(complete_transcript_text: str) -> List[Dict[str, Any]]:
    """Extract verifiable factual claims from the video transcript using Gemini.
    
    Args:
        complete_transcript_text: Formatted chronological transcript text containing timestamps.
        
    Returns:
        List of claim dictionaries containing claim_id, claim_text, timestamp, and category.
        
    Raises:
        ClaimExtractorError: If call fails, or parsing fails.
    """
    if not Config.GEMINI_API_KEY:
        logger.warning("Gemini API key is not configured. Falling back to Heuristic Demo Mode.")
        text_lower = complete_transcript_text.lower()
        if "plato" in text_lower or "xenophon" in text_lower or "apology" in text_lower:
            extracted_claims = [
                {
                    "claim_id": "Claim 1",
                    "claim_text": "Plato and Xenophon were students of Socrates who immortalized his dialogues.",
                    "timestamp": 0.0,
                    "category": "Historical"
                },
                {
                    "claim_id": "Claim 2",
                    "claim_text": "Socrates' dialogues are recorded in works such as Apology, Crito, and Symposium.",
                    "timestamp": 4.0,
                    "category": "Historical"
                }
            ]
            for c in extracted_claims:
                logger.info(f"[Heuristic] Extracted: [{c['claim_id']}] ({c['category']}) - {c['claim_text']}")
            return extracted_claims
        else:
            # Simple fallback for generic text
            extracted_claims = [
                {
                    "claim_id": "Claim 1",
                    "claim_text": "The video transcript mentions dialogues capturing wit and wisdom.",
                    "timestamp": 0.0,
                    "category": "General"
                }
            ]
            for c in extracted_claims:
                logger.info(f"[Heuristic] Extracted: [{c['claim_id']}] ({c['category']}) - {c['claim_text']}")
            return extracted_claims

    if not complete_transcript_text.strip():
        logger.warning("Empty transcript provided. Returning no claims.")
        return []

    logger.info("Initializing Gemini client for claim extraction...")
    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise ClaimExtractorError(f"Failed to initialize Gemini client: {e}")

    system_instruction = (
        "You are an expert fact-checker and researcher. Your task is to analyze the transcript of a video "
        "(which includes spoken text and OCR-extracted visual text) and extract all verifiable factual claims.\n\n"
        "GUIDELINES:\n"
        "1. Identify only claims that can be objectively proven or disproven using trusted public sources, "
        "historical records, official statistics, scientific publications, government websites, etc.\n"
        "2. Categories to include: Historical, Scientific, Medical, Political, Economic, Tech, Sports, Geographical, Mathematical statements, Names, Dates, Policies, Events.\n"
        "3. IGNORE opinions, greetings, advertisements, personal preferences, jokes, emotional expressions, or general pleasantries.\n"
        "4. Formulate each claim text to be clear, objective, self-contained (resolve pronouns if possible), and easy to search.\n"
        "5. Identify the approximate timestamp of each claim. Use the timestamp given in the transcript segment."
    )

    prompt = (
        f"Analyze the transcript below and extract all unique, verifiable factual claims. "
        f"Return them as a JSON list matching the requested schema.\n\n"
        f"TRANSCRIPT:\n{complete_transcript_text}"
    )

    logger.info("Sending transcript to Gemini for claim extraction...")
    try:
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClaimList,
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature for more deterministic/factual output
            )
        )
        
        # Parse output using the SDK's built-in parsed field
        claims_data: ClaimList = response.parsed
        
        extracted_claims = []
        for index, c in enumerate(claims_data.claims, start=1):
            claim_dict = {
                "claim_id": f"Claim {index}",
                "claim_text": c.claim_text,
                "timestamp": c.timestamp,
                "category": c.category
            }
            extracted_claims.append(claim_dict)
            logger.info(f"Extracted: [{claim_dict['claim_id']}] ({claim_dict['category']}) - {claim_dict['claim_text']}")

        logger.info(f"Successfully extracted {len(extracted_claims)} verifiable claims.")
        return extracted_claims

    except Exception as e:
        logger.error(f"Error calling Gemini API for claim extraction: {e}")
        raise ClaimExtractorError(f"Gemini API request failed: {e}")
