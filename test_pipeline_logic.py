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
            confidence=0.98,
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
        self.assertEqual(res["confidence"], 0.98)
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
                    "verification_response": "True",
                    "verdict": "True",
                    "confidence": 0.98,
                    "evidence_summary": "Summary"
                }
            }
        }
        inserted_id = db.insert_video(document)
        
        # Assertions
        self.assertEqual(inserted_id, str(dummy_id))
        mock_collection.insert_one.assert_called_once_with(document)
        db.close()

if __name__ == "__main__":
    unittest.main()
