import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the project root directory to python path if run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reconfigure stdout and stderr to UTF-8 for Windows console support
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # Fallback for old python versions if reconfigure is not available
    pass

from config import Config
from database.mongo_client import MongoDatabase
from utils.video import validate_video_file, get_video_metadata, VideoMetadataError
from utils.clean import remove_repeated_speech
from pipeline.audio_extractor import extract_audio, AudioExtractionError
from pipeline.transcriber import transcribe_audio, TranscriberError
from pipeline.frame_extractor import extract_frames, FrameExtractionError, cleanup_frames_dir
from pipeline.ocr_engine import OCREngine, OCREngineError
from pipeline.merger import merge_speech_and_ocr, compile_complete_text
from pipeline.claim_extractor import extract_claims, extract_claims_from_lyrics, extract_claims_second_pass, extract_ocr_visual_claims, ClaimExtractorError
from pipeline.verifier import verify_claims, VerifierError
from pipeline.content_classifier import classify_video, generate_intelligent_fallback_response, VideoAnalysis, ContentClassifierError
from pipeline.context_reconstructor import reconstruct_global_context, reconstruct_multimodal_context

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("factlens_processing.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("FactLensOrchestrator")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="FactLens: Advanced Video Text & Speech Metadata Extractor"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (supports mp4, mov, avi, mkv, webm)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=Config.FRAME_INTERVAL,
        help=f"Frame extraction interval in seconds (default: {Config.FRAME_INTERVAL})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.OCR_THRESHOLD,
        help=f"EasyOCR confidence threshold (default: {Config.OCR_THRESHOLD})"
    )
    return parser.parse_args()

def process_video_pipeline(video_path_str: str, interval: float, threshold: float) -> Dict[str, Any]:
    """Execute the complete FactLens video processing pipeline.
    
    Args:
        video_path_str: Path to the input video.
        interval: Frame extraction interval in seconds.
        threshold: OCR confidence threshold.
        
    Returns:
        Dict: Complete document that was generated and prepared for database insertion.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting FactLens pipeline for: {video_path_str}")
    logger.info("=" * 60)

    # 0. Setup directories
    Config.setup_temp_dir()
    
    # 1. Video Loading and Validation (Feature 1)
    try:
        video_path = validate_video_file(video_path_str, Config.SUPPORTED_EXTENSIONS)
        metadata = get_video_metadata(video_path)
    except FileNotFoundError as e:
        logger.error(f"Validation failed - File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation failed - Unsupported format: {e}")
        sys.exit(1)
    except VideoMetadataError as e:
        logger.error(f"Validation failed - Video corrupted or unreadable: {e}")
        sys.exit(1)
        
    # 2. Audio Extraction (Feature 2)
    temp_audio_path = None
    try:
        temp_audio_path = extract_audio(video_path, Config.TEMP_FOLDER)
    except AudioExtractionError as e:
        logger.error(f"Pipeline stalled during Audio Extraction: {e}")
        # Note: We continue without audio if audio extraction fails, or we can choose to fail.
        # Clean architecture dictates we raise or fail gracefully depending on criticality.
        # Since transcription relies on audio, this is a blocker.
        sys.exit(1)

    # 3. Speech Recognition (Feature 3)
    speech_segments: List[Dict[str, Any]] = []
    detected_language = "unknown"
    if temp_audio_path and temp_audio_path.exists():
        try:
            raw_speech, detected_language = transcribe_audio(
                audio_path=temp_audio_path,
                model_size=Config.WHISPER_MODEL,
                device="cpu"
            )
            # Text Cleaning: Remove repeated speech segments (Feature 13)
            speech_segments = remove_repeated_speech(raw_speech)
        except TranscriberError as e:
            logger.error(f"Speech recognition step failed: {e}")
            # Whisper failure shouldn't necessarily block OCR if we want partial results,
            # but per instructions, we handle it as a major failure.
            speech_segments = []

    # 4. Frame Extraction (Feature 4)
    frame_items: List[Dict[str, Any]] = []
    try:
        frame_items = extract_frames(
            video_path=video_path,
            output_dir=Config.TEMP_FOLDER,
            interval_seconds=interval
        )
    except FrameExtractionError as e:
        logger.error(f"Frame extraction failed: {e}")
        frame_items = []

    # 5. Visual Text OCR (Feature 5)
    ocr_results: List[Dict[str, Any]] = []
    if frame_items:
        try:
            ocr_engine = OCREngine(languages=Config.OCR_LANGUAGES, threshold=threshold)
            ocr_results = ocr_engine.process_frames(frame_items)
        except OCREngineError as e:
            logger.error(f"OCR step failed: {e}")
            ocr_results = []

    # Clean up extracted audio file
    if temp_audio_path and temp_audio_path.exists():
        try:
            temp_audio_path.unlink()
            logger.info("Temporary audio WAV file deleted.")
        except Exception as e:
            logger.warning(f"Could not delete temporary audio file: {e}")

    # 6. Merge Speech and OCR Results (Feature 6)
    merged_transcript = merge_speech_and_ocr(speech_segments, ocr_results)
    
    # 7. Compile Final Unified Text
    complete_text = compile_complete_text(merged_transcript)
    
    # Context Reconstruction Layer (New Feature)
    logger.info("Executing Context Reconstruction Layer...")
    global_context = {}
    try:
        global_context = reconstruct_global_context(complete_text, metadata)
    except Exception as e:
        logger.error(f"Global context reconstruction failed: {e}")
        global_context = {
            "reconstructed_story": "Context reconstruction failed.",
            "resolved_entities": [],
            "timeline_events": [],
            "overall_narrative": "Failed to parse narrative."
        }

    # Multimodal Context Fusion
    logger.info("Executing Multimodal Context Fusion...")
    multimodal_context = {}
    try:
        multimodal_context = reconstruct_multimodal_context(
            reconstructed_transcript_context=global_context,
            frame_items=frame_items,
            ocr_results=ocr_results
        )
    except Exception as e:
        logger.error(f"Multimodal context fusion failed: {e}")
        multimodal_context = {
            "image_understanding_summary": "Multimodal fusion failed.",
            "multimodal_narrative": "Failed to fuse multimodal elements.",
            "resolved_visual_entities": []
        }
    
    # 8. Content Classification (New Feature)
    logger.info("Executing Content Classification stage...")
    overall_classification = {}
    is_mixed = False
    segments_data = []
    global_claims = {}
    claim_counter = 1
    fact_verification_executed = False
    context_aware_claims_list = []
    all_context_aware_queries = []
    
    try:
        analysis: VideoAnalysis = classify_video(complete_text, metadata)
        is_mixed = analysis.is_mixed_content
        
        # Determine overall classification details
        if analysis.segments:
            overall_classification = {
                "primary_category": "Mixed content" if is_mixed else analysis.segments[0].primary_category,
                "secondary_category": None if is_mixed else analysis.segments[0].secondary_category,
                "confidence": sum(s.confidence for s in analysis.segments) / len(analysis.segments),
                "reason": "Mixed content video with multiple segments." if is_mixed else analysis.segments[0].reason
            }
            
        # Process each segment independently
        for seg in analysis.segments:
            logger.info(f"Processing segment {seg.segment_id} ({seg.start_time}s - {seg.end_time}s) of type: {seg.content_type}")
            
            # Extract transcript subsegment
            seg_entries = [
                entry for entry in merged_transcript
                if seg.start_time <= entry["timestamp"] <= seg.end_time
            ]
            seg_text = compile_complete_text(seg_entries)
            
            seg_claims = []
            seg_verifications = []
            verified = False
            
            if seg.content_type == "informational":
                # Informational: Run Claim Extraction & Verification
                try:
                    seg_claims = extract_claims(seg_text, global_context.get("reconstructed_story"))
                except Exception as e:
                    logger.error(f"Claim extraction failed for segment {seg.segment_id}: {e}")
                
                # Retry second-pass extraction for educational/news categories if first pass returned empty
                is_edu_or_news = any(kw in seg.primary_category.lower() for kw in ["educational", "news", "lecture", "explanation"])
                if not seg_claims and is_edu_or_news:
                    logger.info(f"Segment {seg.segment_id} classified as Educational/News but no claims extracted. Retrying claim extraction with strict second-pass...")
                    try:
                        seg_claims = extract_claims_second_pass(seg_text, global_context.get("reconstructed_story"))
                    except Exception as e:
                        logger.error(f"Second-pass claim extraction failed for segment {seg.segment_id}: {e}")
                
                if seg_claims:
                    fact_verification_executed = True
                    verified = True
                    try:
                        seg_verifications = verify_claims(seg_claims, global_context.get("reconstructed_story"))
                    except Exception as e:
                        logger.error(f"Verification failed for segment {seg.segment_id}: {e}")
                        
            elif seg.content_type == "music":
                # Music: Run selective Lyrics claim extraction
                try:
                    seg_claims = extract_claims_from_lyrics(seg_text, start_time=seg.start_time, reconstructed_context=global_context.get("reconstructed_story"))
                except Exception as e:
                    logger.error(f"Lyrics claim extraction failed for segment {seg.segment_id}: {e}")
                    
                if seg_claims:
                    fact_verification_executed = True
                    verified = True
                    try:
                        # Rename lyric claims to prevent naming collisions
                        for idx, c in enumerate(seg_claims, start=1):
                            c["claim_id"] = f"Claim {idx}"
                        seg_verifications = verify_claims(seg_claims, global_context.get("reconstructed_story"))
                    except Exception as e:
                        logger.error(f"Lyrics claim verification failed for segment {seg.segment_id}: {e}")
            
            # OCR-Based Visual Claim Extraction (Additional Feature)
            ocr_claims = []
            for ocr in ocr_results:
                if seg.start_time <= ocr["timestamp"] <= seg.end_time:
                    # Find matching frame item from the frames list
                    matching_frame = next(
                        (f for f in frame_items if abs(f["timestamp"] - ocr["timestamp"]) < 0.1),
                        None
                    )
                    if matching_frame:
                        try:
                            visual_claims = extract_ocr_visual_claims(
                                frame_path=Path(matching_frame["path"]),
                                ocr_text=ocr["text"],
                                timestamp=ocr["timestamp"],
                                multimodal_context=multimodal_context
                            )
                            ocr_claims.extend(visual_claims)
                        except Exception as ex:
                            logger.error(f"OCR visual claim extraction failed at {ocr['timestamp']}s: {ex}")
            
            if ocr_claims:
                logger.info(f"Adding {len(ocr_claims)} OCR-derived visual claims to segment {seg.segment_id}...")
                
                # Verify these new OCR claims
                fact_verification_executed = True
                verified = True
                try:
                    # Rename ocr claims to avoid conflicts
                    for idx, c in enumerate(ocr_claims, start=len(seg_claims) + 1):
                        c["claim_id"] = f"Claim {idx}"
                    ocr_verifications = verify_claims(ocr_claims, global_context.get("reconstructed_story"))
                    seg_verifications.extend(ocr_verifications)
                except Exception as e:
                    logger.error(f"OCR visual claims verification failed: {e}")
                
                seg_claims.extend(ocr_claims)
            
            # Format segment claims if claims were found
            formatted_seg_claims = {}
            if seg_claims:
                for claim in seg_claims:
                    claim_key = f"Claim {claim_counter}"
                    claim_counter += 1
                    
                    verification = next((v for v in seg_verifications if v["claim_text"] == claim["claim_text"]), None)
                    if verification:
                        formatted_seg_claims[claim_key] = {
                            "original_claim": claim["claim_text"],
                            "search_query_used": verification.get("search_queries", [claim["claim_text"]]),
                            "retrieved_sources": verification.get("sources", []),
                            "evidence_summary": verification.get("evidence_summary", ""),
                            "verification_explanation": verification.get("explanation", ""),
                            "verdict": verification.get("verdict", ""),
                            "confidence": verification.get("confidence", 0.0),
                            "processing_time": verification.get("processing_time", "0.0s")
                        }
                        global_claims[claim_key] = formatted_seg_claims[claim_key]
                        
                        context_aware_claims_list.append({
                            "claim_text": claim["claim_text"],
                            "timestamp": claim["timestamp"],
                            "category": claim.get("category", "General"),
                            "verdict": verification.get("verdict", "")
                        })
                        all_context_aware_queries.extend(verification.get("search_queries", []))
                    else:
                        formatted_seg_claims[claim_key] = {
                            "original_claim": claim["claim_text"],
                            "search_query_used": [claim["claim_text"]],
                            "retrieved_sources": [],
                            "evidence_summary": "No search results retrieved.",
                            "verification_explanation": "Verification skipped or failed.",
                            "verdict": "Needs Human Verification",
                            "confidence": 0.0,
                            "processing_time": "0.0s"
                        }
                        global_claims[claim_key] = formatted_seg_claims[claim_key]
                        
                        context_aware_claims_list.append({
                            "claim_text": claim["claim_text"],
                            "timestamp": claim["timestamp"],
                            "category": claim.get("category", "General"),
                            "verdict": "Needs Human Verification"
                        })
                        all_context_aware_queries.append(claim["claim_text"])
            
            # If no claims were extracted, generate intelligent fallback response
            fallback_info = None
            category_metadata = {}
            if not seg_claims:
                logger.info(f"No claims found for segment {seg.segment_id}. Generating intelligent fallback response...")
                fallback_info = generate_intelligent_fallback_response(
                    transcript_text=seg_text,
                    category=seg.primary_category,
                    confidence=seg.confidence,
                    reason=seg.reason,
                    metadata=metadata
                )
                
                # Build metadata dictionary depending on category
                cat_lower = seg.primary_category.lower()
                if seg.content_type == "music" or any(kw in cat_lower for kw in ["music", "song", "lyrics", "concert", "performance", "remix"]):
                    category_metadata = {
                        "song_title": fallback_info.song_title,
                        "artist": fallback_info.artist,
                        "singer": fallback_info.singer,
                        "composer": fallback_info.composer,
                        "album": fallback_info.album,
                        "release_year": fallback_info.release_year,
                        "genre": fallback_info.genre,
                        "language": fallback_info.language,
                        "music_label": fallback_info.music_label,
                        "duration": fallback_info.duration or f"{seg.end_time - seg.start_time:.2f}s",
                        "popularity_info": fallback_info.popularity_info,
                        "official_version": fallback_info.official_version,
                        "lyrics_factual_claims_detected": fallback_info.lyrics_factual_claims_detected or False
                    }
                elif seg.content_type == "entertainment" or any(kw in cat_lower for kw in ["meme", "comedy", "stand-up", "funny", "reaction", "movie", "tv", "game", "gaming", "highlight", "entertainment"]):
                    category_metadata = {
                        "meme_category": fallback_info.meme_category or seg.primary_category,
                        "humor_type": fallback_info.humor_type or "Fictional / Farcical",
                        "short_description": fallback_info.summary,
                        "reason_no_verification_required": fallback_info.reason
                    }
                elif "podcast" in cat_lower:
                    category_metadata = {
                        "topic": fallback_info.podcast_topic or "Discussion",
                        "summary": fallback_info.summary,
                        "reason_no_factual_claims_detected": "Discussion mainly consists of subjective opinions, experiences, or conversational talk rather than objective facts."
                    }
                elif "interview" in cat_lower:
                    category_metadata = {
                        "topic": fallback_info.interview_topic or "Interview",
                        "summary": fallback_info.summary,
                        "participants": fallback_info.interview_participants or "Host & Guest",
                        "reason_no_factual_claims_detected": "Subjective dialogue and personal experiences."
                    }
                elif any(kw in cat_lower for kw in ["advertisement", "promo", "commercial", "ad"]):
                    category_metadata = {
                        "brand": fallback_info.brand or "Unknown Brand",
                        "product": fallback_info.product or "Unknown Product",
                        "company": fallback_info.company or "Unknown Company",
                        "campaign_slogan": fallback_info.campaign_slogan,
                        "marketing_claims": [fallback_info.summary]
                    }
                elif "unknown" in cat_lower:
                    category_metadata = {
                        "reason": "Insufficient information for reliable classification.",
                        "verification_performed": False
                    }
                else:
                    # General / Educational / News fallback
                    category_metadata = {
                        "topic": seg.primary_category,
                        "summary": fallback_info.summary,
                        "reason_no_factual_claims_detected": fallback_info.reason
                    }

            # Construct segment record
            seg_record = {
                "segment_id": seg.segment_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "primary_category": seg.primary_category,
                "secondary_category": seg.secondary_category,
                "confidence": seg.confidence,
                "reason": seg.reason,
                "content_type": seg.content_type,
                "verified": verified
            }
            
            if seg_claims:
                seg_record["claims"] = formatted_seg_claims
            else:
                seg_record["fallback_response"] = {
                    "explanation": fallback_info.explanation,
                    "summary": fallback_info.summary,
                    "category_metadata": category_metadata
                }
                
            segments_data.append(seg_record)
            
    except Exception as e:
        logger.error(f"Content classification stage failed: {e}")
        # Build simple fallback informational segment so pipeline continues
        is_mixed = False
        overall_classification = {
            "primary_category": "Unknown",
            "secondary_category": None,
            "confidence": 0.0,
            "reason": f"Classification error: {e}"
        }
        
    processing_time = round(time.time() - start_time, 2)
    
    # Construct final document structure for MongoDB (Step 13 extended)
    document = {
        # Video metadata (Step 1)
        "filename": metadata["filename"],
        "duration": metadata["duration"],
        "fps": metadata["fps"],
        "resolution": metadata["resolution"],
        "codec": metadata["codec"],
        "size_bytes": metadata["size_bytes"],
        "size_mb": metadata["size_mb"],
        "upload_timestamp": datetime.fromtimestamp(os.path.getmtime(metadata["path"])).isoformat() + "Z",
        
        # Context Reconstruction database storage
        "context_reconstruction": {
            "reconstructed_story": global_context.get("reconstructed_story"),
            "resolved_entities": global_context.get("resolved_entities", []),
            "timeline_events": global_context.get("timeline_events", []),
            "overall_narrative": global_context.get("overall_narrative", ""),
            "context_aware_search_queries": list(set(all_context_aware_queries)),
            "linked_transcript_chunks": merged_transcript,
            "linked_ocr_frames": ocr_results,
            "image_understanding_summary": multimodal_context.get("image_understanding_summary", ""),
            "multimodal_context": multimodal_context,
            "generated_context_aware_claims": context_aware_claims_list
        },
        
        # Complete merged transcript (Step 6)
        "complete_merged_transcript": complete_text,
        
        # Classification metadata
        "classification": {
            "primary_category": overall_classification.get("primary_category"),
            "secondary_category": overall_classification.get("secondary_category"),
            "confidence": overall_classification.get("confidence", 0.0),
            "reason": overall_classification.get("reason"),
            "is_mixed_content": is_mixed,
            "fact_verification_executed": fact_verification_executed
        },
        
        # Segments list (Mixed content format)
        "segments": segments_data,
        
        # Merged list of claims (Informational format)
        "claims": global_claims,
        
        # Processing timestamps
        "processing_timestamps": {
            "started_at": datetime.fromtimestamp(start_time).isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_time": f"{processing_time}s"
        },
        
        # Model information
        "model_info": {
            "transcription_model": f"faster-whisper-{Config.WHISPER_MODEL}",
            "classification_model": Config.GEMINI_MODEL,
            "claim_extraction_model": Config.GEMINI_MODEL,
            "verification_model": Config.GEMINI_MODEL
        }
    }
    
    # Set overall document fields for fallbacks / category details (prompt spec)
    document["video_category"] = overall_classification.get("primary_category")
    document["classification_confidence"] = overall_classification.get("confidence", 0.0)
    document["reason_for_classification"] = overall_classification.get("reason")
    document["fact_verification_executed"] = fact_verification_executed
    
    # Store fallback responses list at top level
    fallback_list = [
        seg["fallback_response"] for seg in segments_data if "fallback_response" in seg
    ]
    if fallback_list:
        document["fallback_responses"] = fallback_list
        
    if not fact_verification_executed:
        document["reason_for_skipping"] = overall_classification.get("reason")
        if fallback_list:
            document["category_specific_response"] = fallback_list[0]["explanation"]
            document["summary"] = fallback_list[0]["summary"]
            document["additional_metadata"] = fallback_list[0]["category_metadata"]
            
            # Flatten category metadata into the top level of the document
            for k, v in fallback_list[0]["category_metadata"].items():
                document[k] = v
        else:
            document["category_specific_response"] = "No factual claims extracted or verified."
            document["summary"] = "No factual claims detected."
            document["additional_metadata"] = {}
            
    # Clean up temporary frames folder at the end of the pipeline
    frames_dir = Config.TEMP_FOLDER / "frames"
    if frames_dir.exists():
        try:
            cleanup_frames_dir(frames_dir)
            logger.info("Temporary frames directory deleted.")
        except Exception as e:
            logger.warning(f"Could not delete temporary frames directory: {e}")

    return document

def main() -> None:
    """Main execution orchestrator."""
    args = parse_arguments()
    
    # Process the video pipeline
    document = process_video_pipeline(
        video_path_str=args.video_path,
        interval=args.interval,
        threshold=args.threshold
    )
    
    # Persist to MongoDB (Step 13)
    db_id = "Not Inserted"
    db = MongoDatabase(uri=Config.MONGO_URI, db_name=Config.DB_NAME, collection_name=Config.COLLECTION_NAME)
    try:
        db.connect()
        db_id = db.insert_video(document)
    except ConnectionError as e:
        logger.error(f"Database insertion skipped: MongoDB disconnected or unavailable. Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing to MongoDB: {e}")
    finally:
        db.close()

    # Output Statistics and Verified Claims display (Feature 18)
    print("\n" + "=" * 60)
    print("                 FACTLENS PIPELINE SUMMARY                ")
    print("=" * 60)
    print(f"Video File:       {document['filename']}")
    print(f"Resolution:       {document['resolution']}")
    print(f"FPS:              {document['fps']}")
    print(f"Duration:         {document['duration']} seconds")
    print(f"Codec:            {document['codec']}")
    print(f"Size:             {document['size_mb']} MB")
    print(f"Processing Time:  {document['processing_timestamps']['processing_time']}")
    
    cl = document.get("classification", {})
    print(f"Category:         {cl.get('primary_category')} (Confidence: {cl.get('confidence', 0.0):.2f})")
    print(f"Reason:           {cl.get('reason')}")
    print(f"Mixed Content:    {cl.get('is_mixed_content')}")
    print(f"Fact-Checking:    {'Executed' if cl.get('fact_verification_executed') else 'Bypassed'}")
    print(f"MongoDB ObjectId: {db_id}")
    print("=" * 60)
    
    # Print segments if mixed
    if cl.get('is_mixed_content'):
        print("\nSegment Timeline Details:")
        print("-" * 60)
        for seg in document.get("segments", []):
            print(f"Segment {seg['segment_id']} [{seg['start_time']}s - {seg['end_time']}s]:")
            print(f"  Category:  {seg['primary_category']} (Type: {seg['content_type']})")
            print(f"  Reason:    {seg['reason']}")
            if "fallback_response" in seg:
                fallback = seg["fallback_response"]
                print(f"  Summary:   {fallback['summary']}")
                print(f"  Response:  {fallback['explanation']}")
                meta = fallback.get("category_metadata") or {}
                if meta:
                    for k, v in meta.items():
                        if v is not None:
                            print(f"    {k.replace('_', ' ').title()}: {v}")
            print("-" * 60)
            
    # Print overall claims
    claims_dict = document.get("claims", {})
    if claims_dict:
        print(f"\nExtracted & Verified Claims ({len(claims_dict)}):")
        print("-" * 60)
        for c_id, c_val in claims_dict.items():
            print(f"\n[{c_id}]")
            print(f"  Claim:      {c_val['original_claim']}")
            print(f"  Verdict:    {c_val['verdict']} (Confidence: {c_val['confidence']})")
            print(f"  Summary:    {c_val['evidence_summary']}")
            reasoning = c_val.get('verification_explanation') or c_val.get('verification_response', '')
            print(f"  Reasoning:  {reasoning}")
            print("-" * 60)
    else:
        # Check fallback response from the document
        explanation = document.get("category_specific_response")
        summary = document.get("summary")
        category = document.get("video_category")
        reason = document.get("reason_for_classification")
        
        if explanation:
            print("\nIntelligent Fallback Response:")
            print("-" * 60)
            print(f"Category:     {category}")
            print(f"Reason:       {reason}")
            print(f"Response:     {explanation}")
            print(f"Summary:      {summary}")
            
            meta = document.get("additional_metadata") or {}
            if meta:
                print("Additional Metadata:")
                for k, v in meta.items():
                    if v is not None:
                        display_key = k.replace("_", " ").title()
                        print(f"  {display_key}: {v}")
            print("-" * 60)
        else:
            print("\nNo factual claims extracted or verified.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
