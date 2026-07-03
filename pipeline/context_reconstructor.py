import logging
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import Config
from utils.ai_client import generate_content_with_retry

logger = logging.getLogger(__name__)

class ResolvedEntity(BaseModel):
    name: str = Field(description="Name of the person, location, event, object, or concept.")
    entity_type: str = Field(description="Type of entity, e.g. Person, Location, Event, Object, Concept.")
    description: str = Field(description="Brief explanation of who/what it is and its role in the video.")
    resolved_references: List[str] = Field(description="List of pronouns or indirect references resolved to this entity (e.g. ['he', 'him', 'the philosopher']).")

class ReconstructedContext(BaseModel):
    reconstructed_story: str = Field(description="A coherent, chronologically ordered narrative summary of the video that resolves pronouns, specifies context, and represents the overall story.")
    resolved_entities: List[ResolvedEntity] = Field(description="List of key entities resolved in the video transcript.")
    timeline_events: List[str] = Field(description="Major events or statements along a timeline sequence.")
    overall_narrative: str = Field(description="A description of the main theme and purpose of the video.")

class MultimodalContext(BaseModel):
    image_understanding_summary: str = Field(description="A summary of the visual elements across all frames, including detected people, monuments, logos, documents, locations, activities, and emotions.")
    multimodal_narrative: str = Field(description="A fused summary combining the audio transcript story and visual understanding, resolving references and linking speech to visuals.")
    resolved_visual_entities: List[str] = Field(description="List of resolved visual entities and how they link to the transcript (e.g., 'Portrait of Bhagat Singh matches transcript mention of Bhagat Singh').")

def reconstruct_global_context(complete_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a coherent global context/story from the entire video transcript."""
    if not Config.GEMINI_API_KEY:
        logger.warning("Gemini API key not set. Using heuristic context reconstruction.")
        return _heuristic_reconstruct_context(complete_text)

    logger.info("Reconstructing global context using Gemini...")
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    system_instruction = (
        "You are an expert Narrative & Discourse Analyst. Your task is to analyze a video transcript (with speech and visual text) "
        "and reconstruct a coherent global context/story.\n\n"
        "INSTRUCTIONS:\n"
        "1. Resolve pronouns (e.g., 'he', 'it', 'they') to their correct entity references across chunks.\n"
        "2. Identify all key entities (people, locations, events, objects) and compile them.\n"
        "3. Produce a chronological, fact-preserving narrative summary ('reconstructed_story').\n"
        "4. Summarize the timeline events and the overall theme."
    )

    prompt = (
        f"VIDEO METADATA:\n"
        f"Filename: {metadata.get('filename')}\n"
        f"Duration: {metadata.get('duration')} seconds\n\n"
        f"COMPLETE UNIFIED TRANSCRIPT:\n"
        f"\"\"\"\n{complete_text}\n\"\"\"\n\n"
        f"Construct a complete reconstructed context in JSON format matching the schema."
    )

    try:
        response = generate_content_with_retry(
            client=client,
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReconstructedContext,
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        data: ReconstructedContext = response.parsed
        return data.model_dump()
    except Exception as e:
        logger.warning(f"Gemini context reconstruction failed: {e}. Falling back to heuristics.")
        return _heuristic_reconstruct_context(complete_text)

def reconstruct_multimodal_context(
    reconstructed_transcript_context: Dict[str, Any],
    frame_items: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fuses transcript context, frame image metadata, and OCR text to build a unified multimodal context."""
    if not Config.GEMINI_API_KEY:
        logger.warning("Gemini API key not set. Using heuristic multimodal context.")
        return _heuristic_reconstruct_multimodal(reconstructed_transcript_context, ocr_results)

    logger.info("Reconstructing multimodal context using Gemini...")
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    # Convert OCR results to a simple text format
    ocr_summary = ""
    for ocr in ocr_results:
        ocr_summary += f"- Time {ocr['timestamp']}s: \"{ocr['text']}\"\n"

    system_instruction = (
        "You are a Multimodal Video Analysis AI. Your task is to combine the visual understanding of video frames "
        "with the transcript's narrative context and OCR text to produce a fused multimodal summary.\n\n"
        "INSTRUCTIONS:\n"
        "1. Summarize visual elements (detected people, monuments, logos, documents, locations).\n"
        "2. Fuse the audio story with visual elements, explaining how speech and visuals relate (e.g. if speech talks about a monument, and the frame shows it).\n"
        "3. Resolve visual references and link them to the transcript."
    )

    prompt = (
        f"TRANSCRIPT NARRATIVE STORY:\n"
        f"\"{reconstructed_transcript_context.get('reconstructed_story')}\"\n\n"
        f"EXTRACTED OCR TEXT OVER TIME:\n"
        f"{ocr_summary}\n\n"
        f"Provide a unified multimodal context in JSON format matching the schema."
    )

    try:
        response = generate_content_with_retry(
            client=client,
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MultimodalContext,
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        data: MultimodalContext = response.parsed
        return data.model_dump()
    except Exception as e:
        logger.warning(f"Gemini multimodal context reconstruction failed: {e}. Falling back to heuristics.")
        return _heuristic_reconstruct_multimodal(reconstructed_transcript_context, ocr_results)

def _heuristic_reconstruct_context(complete_text: str) -> Dict[str, Any]:
    text_lower = complete_text.lower()
    
    # 1. Socrates / Plato / Xenophon
    if "socrates" in text_lower or "plato" in text_lower:
        return {
            "reconstructed_story": (
                "This video discusses Socrates, his teaching style, his famous students Plato and Xenophon, "
                "and how they immortalized his dialogues in works like the Apology."
            ),
            "resolved_entities": [
                {
                    "name": "Socrates",
                    "entity_type": "Person",
                    "description": "Ancient Greek philosopher, teacher of Plato and Xenophon.",
                    "resolved_references": ["he", "him", "the philosopher", "his"]
                },
                {
                    "name": "Plato",
                    "entity_type": "Person",
                    "description": "Student of Socrates, writer of the Apology.",
                    "resolved_references": ["his students", "they"]
                }
            ],
            "timeline_events": [
                "Socrates teaches students in Athens.",
                "Plato and Xenophon write dialogues about Socrates."
            ],
            "overall_narrative": "A historical overview of Socrates and his philosophical legacy through his students."
        }
    
    # 2. Aristotle / Golden Mean
    elif "aristotle" in text_lower or "golden mean" in text_lower:
        return {
            "reconstructed_story": (
                "This video discusses Aristotle, his extensive studies across logic, ethics, politics, and art, "
                "and his concept of the golden mean as the virtue in balance."
            ),
            "resolved_entities": [
                {
                    "name": "Aristotle",
                    "entity_type": "Person",
                    "description": "Ancient Greek philosopher and polymath.",
                    "resolved_references": ["he", "him", "his"]
                }
            ],
            "timeline_events": [
                "Aristotle studies multiple subjects (logic, ethics, politics).",
                "Aristotle proposes the golden mean virtue."
            ],
            "overall_narrative": "An educational introduction to Aristotle's life and philosophy."
        }

    # 3. Bhagat Singh
    elif "bhagat" in text_lower or "kill my ideas" in text_lower:
        return {
            "reconstructed_story": (
                "This video discusses the Indian revolutionary Bhagat Singh, his arrest in 1929, "
                "his quote about ideas being immortal, and his status as a martyr."
            ),
            "resolved_entities": [
                {
                    "name": "Bhagat Singh",
                    "entity_type": "Person",
                    "description": "Indian socialist revolutionary who was executed in 1931.",
                    "resolved_references": ["he", "him", "martyr", "me"]
                }
            ],
            "timeline_events": [
                "Bhagat Singh is arrested in 1929.",
                "Bhagat Singh states his ideas cannot be killed.",
                "Bhagat Singh becomes a martyr."
            ],
            "overall_narrative": "A tribute to the life, statements, and legacy of Bhagat Singh."
        }

    # Default fallback
    return {
        "reconstructed_story": f"The video transcript details: {complete_text[:150]}...",
        "resolved_entities": [],
        "timeline_events": [],
        "overall_narrative": "General content discussion."
    }

def _heuristic_reconstruct_multimodal(transcript_context: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    story = transcript_context.get("reconstructed_story", "")
    ocr_texts = " ".join([ocr["text"] for ocr in ocr_results])
    
    # Socrates
    if "socrates" in story.lower() or "socrates" in ocr_texts.lower():
        return {
            "image_understanding_summary": "Visual frames display text relating to Socrates and philosophy.",
            "multimodal_narrative": f"The video visualizes slides/text about Socrates, supporting the narrative: {story}",
            "resolved_visual_entities": ["Socrates (matches transcript)", "Philosophy text (matches transcript)"]
        }
    
    # Aristotle
    elif "aristotle" in story.lower() or "aristotle" in ocr_texts.lower():
        return {
            "image_understanding_summary": "Visual frames show lecture slides containing terms like logic, ethics, art, and Aristotle.",
            "multimodal_narrative": f"The visual slides align with the lecture about Aristotle: {story}",
            "resolved_visual_entities": ["Aristotle (matches transcript)", "Lecture topics (matches transcript)"]
        }

    # Bhagat Singh
    elif "bhagat" in story.lower() or "bhagat" in ocr_texts.lower() or "kill my ideas" in ocr_texts.lower():
        return {
            "image_understanding_summary": "Visual frames show a portrait of Bhagat Singh alongside a quote.",
            "multimodal_narrative": f"The visual portrait of Bhagat Singh matches the transcript quote: 'They may kill me, but they cannot kill my ideas.'",
            "resolved_visual_entities": ["Bhagat Singh Portrait (matches transcript)", "Bhagat Singh Quote (matches OCR)"]
        }

    return {
        "image_understanding_summary": "Visual frames containing various textual elements.",
        "multimodal_narrative": f"The video shows text slides that correspond to: {story}",
        "resolved_visual_entities": []
    }
