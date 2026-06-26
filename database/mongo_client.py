import logging
from typing import Any, Dict, Optional
from pymongo import MongoClient, ASCENDING, errors
from bson import ObjectId

logger = logging.getLogger(__name__)

class MongoDatabase:
    """MongoDB client wrapper class for persisting video transcripts in RealityChecker.
    
    Provides connection handling, index creation, and CRUD operations.
    """
    
    def __init__(self, uri: str, db_name: str = "RealityChecker", collection_name: str = "video_transcripts") -> None:
        """Initialize the database client.
        
        Args:
            uri: The MongoDB connection string.
            db_name: The target database name.
            collection_name: The collection where video transcripts will be stored.
        """
        self.uri = uri
        self.db_name_str = db_name
        self.collection_name_str = collection_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None

    def connect(self) -> None:
        """Establish a connection to the MongoDB server and initialize indexes."""
        try:
            logger.info(f"Connecting to MongoDB at {self.uri.split('@')[-1]}")  # Hide credentials if any
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            
            # Trigger server selection to verify connection is alive
            self.client.server_info()
            
            self.db = self.client[self.db_name_str]
            self.collection = self.db[self.collection_name_str]
            
            # Create indexes on 'filename' and 'created_at' as requested
            self.collection.create_index([("filename", ASCENDING)])
            self.collection.create_index([("created_at", ASCENDING)])
            logger.info("MongoDB connected and indexes created successfully.")
            
        except errors.ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB (timeout): {e}")
            raise ConnectionError(f"Could not connect to MongoDB server: {e}")
        except Exception as e:
            logger.error(f"Unexpected database connection error: {e}")
            raise

    def insert_video(self, document: Dict[str, Any]) -> str:
        """Insert a video transcript document.
        
        Args:
            document: The transcript dictionary.
            
        Returns:
            The string representation of the inserted document's ObjectId.
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            result = self.collection.insert_one(document)
            inserted_id = str(result.inserted_id)
            logger.info(f"Successfully inserted video transcript. ObjectId: {inserted_id}")
            return inserted_id
        except errors.PyMongoError as e:
            logger.error(f"Database insertion failed: {e}")
            raise

    def find_video(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single video transcript document matching a query.
        
        Args:
            query: The filter query.
            
        Returns:
            The document dict if found, else None.
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        try:
            return self.collection.find_one(query)
        except errors.PyMongoError as e:
            logger.error(f"Database find query failed: {e}")
            raise

    def delete_video(self, query: Dict[str, Any]) -> int:
        """Delete video transcript documents matching a query.
        
        Args:
            query: The filter query.
            
        Returns:
            The number of deleted documents.
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        try:
            result = self.collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} document(s) matching query: {query}")
            return result.deleted_count
        except errors.PyMongoError as e:
            logger.error(f"Database deletion failed: {e}")
            raise

    def update_video(self, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        """Update video transcript documents matching a query.
        
        Args:
            query: The filter query.
            update_data: The update dictionary containing operators like '$set'.
            
        Returns:
            The number of matched and updated documents.
        """
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        try:
            result = self.collection.update_many(query, update_data)
            logger.info(f"Updated {result.modified_count} document(s) matching query: {query}")
            return result.modified_count
        except errors.PyMongoError as e:
            logger.error(f"Database update failed: {e}")
            raise

    def close(self) -> None:
        """Close the MongoDB client connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            logger.info("MongoDB client connection closed.")
