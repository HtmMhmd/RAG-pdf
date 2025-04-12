import logging
import numpy as np
import openai
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.openai_api_key = config.openai_api_key
        self.model = config.embedding_model
        self.use_local_embeddings = config.use_local_embeddings
        self.local_model_name = config.local_embedding_model
        
        # Initialize local model if needed
        self.local_model = None
        if self.use_local_embeddings:
            logger.info(f"Using local embedding model: {self.local_model_name}")
            self.local_model = SentenceTransformer(self.local_model_name)
        
    def generate_embeddings_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        openai.api_key = self.openai_api_key
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1) // batch_size}")
            
            response = openai.embeddings.create(
                model=self.model,
                input=batch_texts
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def generate_embeddings_with_local_model(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using a local sentence transformer model."""
        if self.local_model is None:
            logger.info(f"Initializing local embedding model: {self.local_model_name}")
            self.local_model = SentenceTransformer(self.local_model_name)
        
        logger.info(f"Generating embeddings for {len(texts)} chunks using local model")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1) // batch_size}")
            
            batch_embeddings = self.local_model.encode(batch_texts).tolist()
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks using OpenAI API or local model.
        
        Args:
            chunks: List of dictionaries with text chunks and metadata
            
        Returns:
            List of dictionaries with text chunks, metadata, and embeddings
        """
        try:
            texts = [chunk["content"] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            
            # First check if we're configured to use local embeddings by default
            if self.use_local_embeddings:
                all_embeddings = self.generate_embeddings_with_local_model(texts)
            else:
                # Try with OpenAI first
                try:
                    all_embeddings = self.generate_embeddings_with_openai(texts)
                except Exception as e:
                    logger.warning(f"OpenAI API error: {e}. Falling back to local embedding model.")
                    all_embeddings = self.generate_embeddings_with_local_model(texts)
            
            # Add embeddings to chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                chunk["embedding"] = embedding
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
