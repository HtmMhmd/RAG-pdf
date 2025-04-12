import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, config):
        self.config = config
        self.db_path = config.vector_db_path
        
        # Create the directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Use OpenAI embeddings if we're using their API
        if "ada" in config.embedding_model or "openai" in config.embedding_model.lower():
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.openai_api_key,
                model_name=config.embedding_model
            )
        else:
            # Default to a local embedding model
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
    
    def get_collection_name(self, pdf_path: str) -> str:
        """Generate a unique collection name based on the PDF path."""
        return f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}"
    
    def collection_exists(self, pdf_path: str) -> bool:
        """Check if a collection for the given PDF already exists."""
        collection_name = self.get_collection_name(pdf_path)
        try:
            collections = self.client.list_collections()
            return any(collection.name == collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def create_collection(self, pdf_path: str) -> chromadb.Collection:
        """Create a new collection for the PDF."""
        collection_name = self.get_collection_name(pdf_path)
        try:
            return self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_collection(self, pdf_path: str) -> chromadb.Collection:
        """Get the collection for the PDF."""
        collection_name = self.get_collection_name(pdf_path)
        try:
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            raise
    
    def add_chunks_to_db(self, chunks: List[Dict[str, Any]], pdf_path: str) -> None:
        """
        Add text chunks to the vector database.
        
        Args:
            chunks: List of dictionaries with text chunks, metadata, and embeddings
            pdf_path: Path to the PDF file
        """
        try:
            # Create a new collection or get existing one
            if self.collection_exists(pdf_path):
                logger.info(f"Deleting existing collection for {pdf_path}")
                collection_name = self.get_collection_name(pdf_path)
                self.client.delete_collection(collection_name)
            
            collection = self.create_collection(pdf_path)
            
            # Prepare data for insertion
            ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add documents to the collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to the vector database")
            
        except Exception as e:
            logger.error(f"Error adding chunks to database: {e}")
            raise
    
    def query_db(self, question: str, pdf_path: str, n_results: int = None) -> List[Dict[str, Any]]:
        """
        Query the vector database to find chunks relevant to the question.
        
        Args:
            question: The user's question
            pdf_path: Path to the PDF file
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with their content and metadata
        """
        if n_results is None:
            n_results = self.config.max_chunks
            
        try:
            collection = self.get_collection(pdf_path)
            
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            # Convert results to a more convenient format
            chunks = []
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for the query")
            return chunks
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            raise
