import unittest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from pydantic import BaseModel
from typing import List

# Import our pipeline modules
from config import Config
from pipeline.claim_extractor import extract_claims, Claim, ClaimList, ClaimExtractorError
from pipeline.verifier import verify_single_claim, VerificationResponse, execute_search
from database.mongo_client import MongoDatabase

class TestFactLensPipelineLogic(unittest.TestCase):
    
    def setUp(self):
        # Configure a dummy API key for testing
        Config.GEMINI_API_KEY = "dummy_test_key"
        Config.GEMINI_MODEL = "gemini-2.5-flash"
        
    @patch("pipeline.claim_extractor.genai.Client")
    def test_claim_extraction(self, mock_client_class):
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create a mock response matching ClaimList schema
        mock_response = MagicMock()
        mock_response.parsed = ClaimList(
            claims=[
                Claim(
                    claim_id="Claim 1",
                    claim_text="The Eiffel Tower is 330 meters tall.",
                    timestamp=5.2,
                    category="Geographical"
                ),
                Claim(
                    claim_id="Claim 2",
                    claim_text="The speed of light is 299,792,458 m/s.",
                    timestamp=12.1,
                    category="Scientific"
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response
        
        # Run extraction
        transcript = "[00:00:05] Speech: \"The Eiffel Tower is 330 meters tall.\" | [00:00:12] Speech: \"Light speed is 299792458 meters per second.\""
        claims = extract_claims(transcript)
        
        # Assertions
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["claim_id"], "Claim 1")
        self.assertEqual(claims[0]["claim_text"], "The Eiffel Tower is 330 meters tall.")
        self.assertEqual(claims[0]["timestamp"], 5.2)
        self.assertEqual(claims[0]["category"], "Geographical")
        
        self.assertEqual(claims[1]["claim_id"], "Claim 2")
        self.assertEqual(claims[1]["claim_text"], "The speed of light is 299,792,458 m/s.")
        self.assertEqual(claims[1]["timestamp"], 12.1)
        self.assertEqual(claims[1]["category"], "Scientific")
        
        mock_client.models.generate_content.assert_called_once()

    @patch("pipeline.verifier.execute_search")
    @patch("pipeline.verifier.genai.Client")
    def test_claim_verification(self, mock_client_class, mock_execute_search):
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # 1. Mock search queries response
        from pipeline.verifier import SearchQueries
        mock_query_response = MagicMock()
        mock_query_response.parsed = SearchQueries(queries=["Eiffel Tower height meters", "official height Eiffel Tower"])
        
        # 2. Mock RAG verification response
        mock_verif_response = MagicMock()
        mock_verif_response.parsed = VerificationResponse(
            verdict="True",
            confidence=98.0,
            evidence_summary="According to the official Eiffel Tower website, the tower's height is 330 meters including the antenna.",
            explanation="The claim is correct. Sources verify that the Eiffel Tower is 330 meters tall."
        )
        
        # Configure model calls in order
        mock_client.models.generate_content.side_effect = [mock_query_response, mock_verif_response]
        
        # Mock search execution results
        mock_execute_search.return_value = [
            {
                "title": "Eiffel Tower - Official Website",
                "url": "https://www.toureiffel.paris/en",
                "snippet": "The Eiffel Tower is 330 meters tall, including the new antenna installed in 2022."
            }
        ]
        
        # Run verification
        claim = {
            "claim_id": "Claim 1",
            "claim_text": "The Eiffel Tower is 330 meters tall.",
            "timestamp": 5.2,
            "category": "Geographical"
        }
        res = verify_single_claim(mock_client, claim)
        
        # Assertions
        self.assertEqual(res["claim_id"], "Claim 1")
        self.assertEqual(res["verdict"], "True")
        self.assertEqual(res["confidence"], 98.0)
        self.assertEqual(res["evidence_summary"], "According to the official Eiffel Tower website, the tower's height is 330 meters including the antenna.")
        self.assertIn("https://www.toureiffel.paris/en", res["sources"][0]["url"])

    @patch("database.mongo_client.MongoClient")
    def test_mongodb_insertion(self, mock_mongo_client):
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock insert_one result
        mock_insert_result = MagicMock()
        dummy_id = ObjectId()
        mock_insert_result.inserted_id = dummy_id
        mock_collection.insert_one.return_value = mock_insert_result
        
        # Initialize DB and run insert
        db = MongoDatabase(uri="mongodb://localhost:27017/", db_name="RealityChecker", collection_name="video_transcripts")
        db.connect()
        
        document = {
            "filename": "video.mp4",
            "duration": 17.52,
            "claims": {
                "Claim 1": {
                    "original_claim": "The Eiffel Tower is 330 meters tall.",
                    "search_query_used": ["The Eiffel Tower is 330 meters tall."],
                    "retrieved_sources": [],
                    "evidence_summary": "Summary",
                    "verification_explanation": "Verified.",
                    "verdict": "True",
                    "confidence": 98.0,
                    "processing_time": "1.2s"
                }
            }
        }
        inserted_id = db.insert_video(document)
        
        # Assertions
        self.assertEqual(inserted_id, str(dummy_id))
        mock_collection.insert_one.assert_called_once_with(document)
        db.close()

    @patch("pipeline.content_classifier.genai.Client")
    def test_content_classification(self, mock_client_class):
        from pipeline.content_classifier import classify_video, VideoAnalysis, VideoSegment, SongInfo
        
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = VideoAnalysis(
            is_mixed_content=True,
            segments=[
                VideoSegment(
                    segment_id=1,
                    start_time=0.0,
                    end_time=10.0,
                    primary_category="News",
                    confidence=0.95,
                    reason="News report segment.",
                    content_type="informational"
                ),
                VideoSegment(
                    segment_id=2,
                    start_time=10.0,
                    end_time=20.0,
                    primary_category="Song",
                    confidence=0.90,
                    reason="Music insert segment.",
                    content_type="music",
                    song_info=SongInfo(title="National Anthem", artist="Various"),
                    lyrics_summary="Patriotic lyrics segment."
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response
        
        # Run classification
        metadata = {"filename": "mixed_video.mp4", "duration": 20.0, "resolution": "1920x1080"}
        result = classify_video("Some news text followed by a patriotic song lyric.", metadata)
        
        # Assertions
        self.assertTrue(result.is_mixed_content)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].primary_category, "News")
        self.assertEqual(result.segments[0].content_type, "informational")
        self.assertEqual(result.segments[1].primary_category, "Song")
        self.assertEqual(result.segments[1].content_type, "music")
        self.assertEqual(result.segments[1].song_info.title, "National Anthem")

    @patch("pipeline.claim_extractor.genai.Client")
    def test_lyrics_claim_extraction(self, mock_client_class):
        from pipeline.claim_extractor import extract_claims_from_lyrics
        
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = ClaimList(
            claims=[
                Claim(
                    claim_id="Claim 1",
                    claim_text="India became independent in 1947.",
                    timestamp=0.0,
                    category="Historical"
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response
        
        # Run extraction
        claims = extract_claims_from_lyrics("India became independent in 1947, in the summer night of love.", start_time=12.0)
        
        # Assertions
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim_text"], "India became independent in 1947.")
        # Note: the timestamp matches the start_time parameter passed to extract_claims_from_lyrics
        self.assertEqual(claims[0]["timestamp"], 12.0)
        self.assertEqual(claims[0]["category"], "Historical")

    @patch("pipeline.claim_extractor.genai.Client")
    def test_second_pass_claim_extraction(self, mock_client_class):
        from pipeline.claim_extractor import extract_claims_second_pass
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = ClaimList(
            claims=[
                Claim(
                    claim_id="Claim 1",
                    claim_text="Aristotle studied logic.",
                    timestamp=0.0,
                    category="Historical"
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response
        
        claims = extract_claims_second_pass("He studied logic, ethics, art.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim_text"], "Aristotle studied logic.")
        
    @patch("pipeline.content_classifier.genai.Client")
    def test_intelligent_fallback_response(self, mock_client_class):
        from pipeline.content_classifier import generate_intelligent_fallback_response, IntelligentFallbackResponse
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = IntelligentFallbackResponse(
            category="Music",
            confidence=0.98,
            reason="Heuristic match: Music keywords detected.",
            summary="Artistic song lyrics.",
            explanation="This video has been classified as a music video. The lyrics primarily contain artistic or emotional expressions rather than objectively verifiable factual claims, so no fact verification was required.",
            song_title="Mock Socrates Song",
            artist="Philosopher Band"
        )
        mock_client.models.generate_content.return_value = mock_response
        
        fallback = generate_intelligent_fallback_response(
            transcript_text="philosopher band mock socrates song",
            category="Music",
            confidence=0.98,
            reason="Heuristic match: Music keywords detected.",
            metadata={}
        )
        
        self.assertEqual(fallback.category, "Music")
        self.assertEqual(fallback.song_title, "Mock Socrates Song")
        self.assertEqual(fallback.artist, "Philosopher Band")

    @patch("pipeline.claim_extractor.genai.Client")
    @patch("PIL.Image.open")
    def test_ocr_visual_claim_extraction(self, mock_image_open, mock_client_class):
        from pipeline.claim_extractor import extract_ocr_visual_claims
        
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = ClaimList(
            claims=[
                Claim(
                    claim_id="Claim 1",
                    claim_text="The quote 'They may kill me, but they cannot kill my ideas.' was said by Bhagat Singh.",
                    timestamp=5.0,
                    category="Historical"
                ),
                Claim(
                    claim_id="Claim 2",
                    claim_text="The quote shown in the video is correctly attributed to Bhagat Singh.",
                    timestamp=5.0,
                    category="Historical"
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response
        
        claims = extract_ocr_visual_claims(
            frame_path="mock_frame.jpg",
            ocr_text="They may kill me, but they cannot kill my ideas. - Bhagat Singh",
            timestamp=5.0
        )
        
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["claim_text"], "The quote 'They may kill me, but they cannot kill my ideas.' was said by Bhagat Singh.")
        self.assertEqual(claims[1]["claim_text"], "The quote shown in the video is correctly attributed to Bhagat Singh.")

    @patch("pipeline.context_reconstructor.genai.Client")
    def test_global_context_reconstruction(self, mock_client_class):
        from pipeline.context_reconstructor import reconstruct_global_context, ReconstructedContext, ResolvedEntity
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.parsed = ReconstructedContext(
            reconstructed_story="Socrates teaches students in Athens.",
            resolved_entities=[
                ResolvedEntity(name="Socrates", entity_type="Person", description="Greek philosopher", resolved_references=["he"])
            ],
            timeline_events=["Socrates teaches students"],
            overall_narrative="Philosophical overview"
        )
        mock_client.models.generate_content.return_value = mock_response
        
        ctx = reconstruct_global_context("Socrates teaches students.", {"filename": "socrates.mp4"})
        self.assertEqual(ctx["reconstructed_story"], "Socrates teaches students in Athens.")
        self.assertEqual(ctx["resolved_entities"][0]["name"], "Socrates")
        
    def test_heuristic_context_reconstruction(self):
        from pipeline.context_reconstructor import _heuristic_reconstruct_context
        ctx = _heuristic_reconstruct_context("Socrates was a philosopher.")
        self.assertEqual(ctx["resolved_entities"][0]["name"], "Socrates")
        
    def test_heuristic_pronoun_resolution(self):
        from pipeline.claim_extractor import _resolve_heuristic_pronouns
        resolved = _resolve_heuristic_pronouns("He taught Plato.", "Socrates was a philosopher.")
        self.assertEqual(resolved, "Socrates taught Plato.")

if __name__ == "__main__":
    unittest.main()
