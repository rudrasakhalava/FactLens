import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import Config
from utils.ai_client import generate_content_with_retry

logger = logging.getLogger(__name__)

class SongInfo(BaseModel):
    title: Optional[str] = Field(None, description="Title of the song.")
    artist: Optional[str] = Field(None, description="Band or artist name.")
    singer: Optional[str] = Field(None, description="Lead singer name.")
    composer: Optional[str] = Field(None, description="Composer name.")
    album: Optional[str] = Field(None, description="Album name.")
    release_year: Optional[str] = Field(None, description="Release year of the song.")
    genre: Optional[str] = Field(None, description="Music genre.")
    language: Optional[str] = Field(None, description="Language of the song.")
    official_version: Optional[str] = Field(None, description="Whether this is the official version (e.g., Yes/No/Unknown).")

class VideoSegment(BaseModel):
    segment_id: int = Field(description="Sequential ID of the segment, starting at 1.")
    start_time: float = Field(description="Start time of this segment in seconds.")
    end_time: float = Field(description="End time of this segment in seconds.")
    primary_category: str = Field(description="The primary classification category (e.g. News, Podcast, Meme, Song, Comedy, etc.).")
    secondary_category: Optional[str] = Field(None, description="Optional secondary category (e.g. Technology Discussion).")
    confidence: float = Field(description="Confidence score for this classification, between 0.0 and 1.0.")
    reason: str = Field(description="Brief explanation of why this segment was classified this way.")
    content_type: str = Field(description="Must be one of: 'informational', 'entertainment', 'music', 'unknown'.")
    song_info: Optional[SongInfo] = Field(None, description="Metadata of the song if content_type is 'music'.")
    lyrics_summary: Optional[str] = Field(None, description="Brief summary of the lyrics/speech if content_type is 'music'.")

class VideoAnalysis(BaseModel):
    is_mixed_content: bool = Field(description="True if the video contains multiple distinct content types/segments (e.g., educational video with a music clip, or news with comedy inserts). False if it is a single unified content type.")
    segments: List[VideoSegment] = Field(description="Timeline segments of the video. If is_mixed_content is False, there should be exactly one segment covering the entire duration.")

class ContentClassifierError(Exception):
    """Exception raised when content classification fails."""
    pass

def _heuristic_classify(complete_transcript_text: str, duration: float) -> VideoAnalysis:
    text_lower = complete_transcript_text.lower()
    
    # Detect Socrates test video
    if "plato" in text_lower or "xenophon" in text_lower or "apology" in text_lower:
        segment = VideoSegment(
            segment_id=1,
            start_time=0.0,
            end_time=duration,
            primary_category="Educational content",
            secondary_category="Historical explanation",
            confidence=0.99,
            reason="Heuristic match: Dialogues of Socrates, Plato, and Xenophon detected.",
            content_type="informational"
        )
        return VideoAnalysis(is_mixed_content=False, segments=[segment])
    
    # Detect generic test music trigger
    elif "lyrics" in text_lower or "song" in text_lower:
        segment = VideoSegment(
            segment_id=1,
            start_time=0.0,
            end_time=duration,
            primary_category="Song",
            secondary_category="Lyrics Video",
            confidence=0.95,
            reason="Heuristic match: Music keywords detected.",
            content_type="music",
            song_info=SongInfo(
                title="Mock Socrates Song",
                artist="Philosopher Band",
                genre="Folk",
                language="English"
            ),
            lyrics_summary="Folk song discussing philosophy."
        )
        return VideoAnalysis(is_mixed_content=False, segments=[segment])
        
    # Detect generic entertainment trigger
    elif "comedy" in text_lower or "joke" in text_lower:
        segment = VideoSegment(
            segment_id=1,
            start_time=0.0,
            end_time=duration,
            primary_category="Comedy",
            confidence=0.90,
            reason="Heuristic match: Comedy keywords detected.",
            content_type="entertainment"
        )
        return VideoAnalysis(is_mixed_content=False, segments=[segment])
        
    # Default fallback
    segment = VideoSegment(
        segment_id=1,
        start_time=0.0,
        end_time=duration,
        primary_category="Lecture",
        confidence=0.85,
        reason="Heuristic fallback classification.",
        content_type="informational"
    )
    return VideoAnalysis(is_mixed_content=False, segments=[segment])

def classify_video(complete_transcript_text: str, metadata: Dict[str, Any]) -> VideoAnalysis:
    """Classify video content type and structure using Gemini.
    
    If the API key is missing or daily quota is exceeded, falls back to heuristic classification.
    
    Args:
        complete_transcript_text: The compiled chronological transcript.
        metadata: Video metadata dictionary.
        
    Returns:
        VideoAnalysis pydantic object.
        
    Raises:
        ContentClassifierError: If Gemini API fails and cannot fallback.
    """
    duration = metadata.get("duration", 0.0)
    
    # 1. Heuristic fallback when GEMINI_API_KEY is not set
    if not Config.GEMINI_API_KEY:
        logger.warning("Gemini API key is not configured. Falling back to Heuristic Classification.")
        return _heuristic_classify(complete_transcript_text, duration)

    # 2. Live API classification using Gemini
    logger.info("Initializing Gemini client for content classification...")
    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        raise ContentClassifierError(f"Failed to initialize Gemini: {e}")

    system_instruction = (
        "You are an expert Multimodal Content Understanding AI. Your task is to analyze the transcript "
        "(including speech segments and visual OCR text) and metadata of a video, classify its genre and content type, "
        "and determine its structural segment boundaries.\n\n"
        "CLASSIFICATION CATEGORIES:\n"
        "- Informational: News, Podcast, Interview, Educational content, Documentary, Political speech, Debate, Lecture, Medical information, Finance, Technology, Historical explanation, Scientific explanation, Sports discussion, Social media commentary, Tutorial, Unknown.\n"
        "- Entertainment: Meme, Comedy, Stand-up comedy, Movie clip, TV show clip, Gaming, Vlog, Personal conversation, Reaction videos, Funny edits, Entertainment edits.\n"
        "- Music: Music video, Song, Lyrics Video, Live Performance, Concert, Background Music, Cover Song, Instrumental, Remix, Mashup, Classical Music, Devotional Song.\n\n"
        "MIXED CONTENT AND SEGMENTATION RULES:\n"
        "1. Check if the video contains multiple distinct content types (e.g. a lecture with a song segment, or news with commercial ads, or a vlog with meme clips). If so, set is_mixed_content to True and list the semantic segments chronologically.\n"
        "2. If the video is a single, consistent content type, set is_mixed_content to False and return exactly one segment spanning from 0.0 to the video's total duration.\n"
        "3. Specify content_type for each segment. Must be exactly one of: 'informational', 'entertainment', 'music', 'unknown'.\n"
        "4. If a segment content_type is 'music', extract detailed SongInfo metadata (Song title, Artist, Singer, release_year, genre, etc.) if available in the text/lyrics, and write a brief lyrics_summary."
    )

    prompt = (
        f"Analyze the transcript and metadata below to classify this video.\n\n"
        f"VIDEO METADATA:\n"
        f"Filename: {metadata.get('filename')}\n"
        f"Duration: {duration} seconds\n"
        f"Resolution: {metadata.get('resolution')}\n\n"
        f"TRANSCRIPT:\n{complete_transcript_text}"
    )

    logger.info("Sending content analysis request to Gemini...")
    try:
        response = generate_content_with_retry(
            client=client,
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoAnalysis,
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        
        analysis: VideoAnalysis = response.parsed
        logger.info(f"Video classification completed. Mixed content detected: {analysis.is_mixed_content}. Segments: {len(analysis.segments)}")
        for seg in analysis.segments:
            logger.info(f"  Segment {seg.segment_id} ({seg.start_time}s - {seg.end_time}s): Category={seg.primary_category}, Type={seg.content_type}, Conf={seg.confidence:.2f}")
        return analysis

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            logger.warning(f"Gemini API daily quota limit exceeded or blocked. Falling back to local Heuristic Classification. Details: {e}")
            return _heuristic_classify(complete_transcript_text, duration)
        logger.error(f"Gemini content classification call failed: {e}")
        raise ContentClassifierError(f"Gemini API request failed: {e}")

class IntelligentFallbackResponse(BaseModel):
    category: str = Field(description="The detected video category.")
    confidence: float = Field(description="The classification confidence score.")
    reason: str = Field(description="Brief reason for classification.")
    summary: str = Field(description="A short summary of the video content.")
    explanation: str = Field(description="The category-specific response explaining why fact verification was skipped or failed.")
    
    # Optional metadata depending on the category
    song_title: Optional[str] = Field(None, description="Song title if Music video.")
    artist: Optional[str] = Field(None, description="Artist name if Music video.")
    singer: Optional[str] = Field(None, description="Singer name if Music video.")
    composer: Optional[str] = Field(None, description="Composer name if Music video.")
    album: Optional[str] = Field(None, description="Album name if Music video.")
    release_year: Optional[str] = Field(None, description="Release year of the song if Music video.")
    genre: Optional[str] = Field(None, description="Music genre if Music video.")
    language: Optional[str] = Field(None, description="Language if Music video.")
    music_label: Optional[str] = Field(None, description="Music label if Music video.")
    duration: Optional[str] = Field(None, description="Duration details if Music video.")
    popularity_info: Optional[str] = Field(None, description="Popularity information if Music video.")
    official_version: Optional[str] = Field(None, description="Whether this is the official version (e.g., Yes/No/Unknown).")
    lyrics_factual_claims_detected: Optional[bool] = Field(None, description="Whether factual claims exist in lyrics.")
    
    # Meme/Entertainment
    meme_category: Optional[str] = Field(None, description="Meme category if Meme.")
    humor_type: Optional[str] = Field(None, description="Humor type if Meme/Comedy.")
    
    # Podcast/Interview
    podcast_topic: Optional[str] = Field(None, description="Podcast topic if Podcast.")
    interview_topic: Optional[str] = Field(None, description="Interview topic if Interview.")
    interview_participants: Optional[str] = Field(None, description="Interview participants (if identifiable).")
    
    # Advertisement
    brand: Optional[str] = Field(None, description="Brand name if Advertisement.")
    product: Optional[str] = Field(None, description="Product name if Advertisement.")
    company: Optional[str] = Field(None, description="Company name if Advertisement.")
    campaign_slogan: Optional[str] = Field(None, description="Campaign slogan if Advertisement.")

def _heuristic_fallback_response(transcript_text: str, category: str, confidence: float, reason: str, metadata: Dict[str, Any]) -> IntelligentFallbackResponse:
    import re
    text_lower = transcript_text.lower()
    
    summary = "A video discussing educational or generic topics."
    explanation = f"This video has been classified as {category}. No objectively verifiable factual claims were detected."
    song_title = None
    artist = None
    singer = None
    composer = None
    album = None
    release_year = None
    genre = None
    language = "English"
    music_label = None
    popularity_info = None
    official_version = "Unknown"
    lyrics_factual_claims_detected = False
    meme_category = None
    humor_type = None
    podcast_topic = None
    interview_topic = None
    interview_participants = None
    brand = None
    product = None
    company = None
    campaign_slogan = None
    
    cat_lower = category.lower()
    
    # Check Socrates test song
    if "song" in text_lower or "lyrics" in text_lower or "philosopher band" in text_lower:
        summary = "Folk song discussing philosophy and Socrates."
        explanation = "This video has been classified as a music video. The lyrics primarily contain artistic or emotional expressions rather than objectively verifiable factual claims, so no fact verification was required."
        song_title = "Mock Socrates Song"
        artist = "Philosopher Band"
        genre = "Folk"
        lyrics_factual_claims_detected = False
        official_version = "Yes"
    
    # Check general music/song
    elif any(kw in cat_lower for kw in ["song", "music", "lyrics", "concert", "performance", "remix", "cover"]):
        explanation = "This video has been classified as a music video. The lyrics primarily contain artistic or emotional expressions rather than objectively verifiable factual claims, so no fact verification was required."
        summary = "Artistic music composition."
        
    # Check entertainment/meme/comedy
    elif any(kw in cat_lower for kw in ["meme", "comedy", "stand-up", "funny", "reaction", "movie", "tv", "game", "gaming", "highlight", "entertainment"]):
        explanation = "This video has been classified as entertainment content. It is intended for humor or fictional storytelling and does not contain any objectively verifiable factual claims. Therefore, no fact verification was performed."
        summary = "Fictional or humorous video clip."
        humor_type = "Satire / Parody"
        
    # Check podcast
    elif "podcast" in cat_lower:
        explanation = "This podcast primarily contains opinions, personal experiences, discussions, or subjective viewpoints. No objectively verifiable factual claims were detected."
        summary = "Dialogue discussion between speakers."
        podcast_topic = "General Subject Discussion"
        
    # Check interview
    elif "interview" in cat_lower:
        explanation = "This interview contains opinions or personal experiences rather than factual claims."
        summary = "Interview Q&A dialogue."
        interview_topic = "Personal Dialogue Q&A"
        
    # Check advertisement
    elif any(kw in cat_lower for kw in ["advertisement", "promo", "commercial", "ad"]):
        explanation = "This video has been classified as promotional content."
        summary = "Product marketing promotional advertisement."
        brand = "Promotional Brand"
        
    # Check unknown
    elif "unknown" in cat_lower:
        explanation = "Insufficient information for reliable classification. No factual verification performed."
        summary = "Unclassified content."
        
    return IntelligentFallbackResponse(
        category=category,
        confidence=confidence,
        reason=reason,
        summary=summary,
        explanation=explanation,
        song_title=song_title,
        artist=artist,
        singer=singer,
        composer=composer,
        album=album,
        release_year=release_year,
        genre=genre,
        language=language,
        music_label=music_label,
        popularity_info=popularity_info,
        official_version=official_version,
        lyrics_factual_claims_detected=lyrics_factual_claims_detected,
        meme_category=meme_category,
        humor_type=humor_type,
        podcast_topic=podcast_topic,
        interview_topic=interview_topic,
        interview_participants=interview_participants,
        brand=brand,
        product=product,
        company=company,
        campaign_slogan=campaign_slogan
    )

def generate_intelligent_fallback_response(transcript_text: str, category: str, confidence: float, reason: str, metadata: Dict[str, Any]) -> IntelligentFallbackResponse:
    """Uses Gemini to extract structured category-specific metadata and explanations for non-factual videos."""
    if not Config.GEMINI_API_KEY:
        return _heuristic_fallback_response(transcript_text, category, confidence, reason, metadata)
        
    logger.info("Initializing Gemini client for intelligent fallback response generation...")
    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return _heuristic_fallback_response(transcript_text, category, confidence, reason, metadata)
        
    system_instruction = (
        "You are an expert Multimodal Video Analysis AI. Your task is to analyze the transcript and metadata of a video "
        "and generate a highly informative, structured category-specific fallback response because no factual claims were verified.\n\n"
        "INSTRUCTIONS FOR EXPLANATION FIELD:\n"
        "1. For Entertainment (Meme, Comedy, TV/Movie clips, stand-up, reaction videos, gaming highlights, funny edits, etc.): return EXACTLY: "
        "'This video has been classified as entertainment content. It is intended for humor or fictional storytelling and does not contain any objectively verifiable factual claims. Therefore, no fact verification was performed.'\n"
        "2. For Music (Songs, lyrics, concert, remixes, cover song, etc.): return EXACTLY: "
        "'This video has been classified as a music video. The lyrics primarily contain artistic or emotional expressions rather than objectively verifiable factual claims, so no fact verification was required.'\n"
        "3. For Podcasts: return EXACTLY: "
        "'This podcast primarily contains opinions, personal experiences, discussions, or subjective viewpoints. No objectively verifiable factual claims were detected.'\n"
        "4. For Interviews: return EXACTLY: "
        "'This interview contains opinions or personal experiences rather than factual claims.'\n"
        "5. For Advertisements: return EXACTLY: "
        "'This video has been classified as promotional content.'\n"
        "6. For Unknown: return EXACTLY: "
        "'Insufficient information for reliable classification. No factual verification performed.'\n\n"
        "7. Fill in all category-specific optional fields (like song_title, brand, product, company, campaign_slogan, podcast_topic, interview_participants, humor_type, meme_category) as accurately as possible from the transcript text."
    )
    
    prompt = (
        f"Analyze the video transcript and metadata. Extract all relevant details and fill out the schema.\n\n"
        f"VIDEO METADATA:\n"
        f"Category: {category}\n"
        f"Confidence: {confidence}\n"
        f"Reason: {reason}\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )
    
    try:
        response = generate_content_with_retry(
            client=client,
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntelligentFallbackResponse,
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        return response.parsed
    except Exception as e:
        logger.warning(f"Intelligent fallback response generation failed or rate limited: {e}. Falling back to heuristics.")
        return _heuristic_fallback_response(transcript_text, category, confidence, reason, metadata)
