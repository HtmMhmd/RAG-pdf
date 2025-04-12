import logging
import numpy as np
import openai
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        openai.api_key = config.openai_api_key
        self.model = config.embedding_model
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks using OpenAI API.
        
        Args:
            chunks: List of dictionaries with text chunks and metadata
            
        Returns:
            List of dictionaries with text chunks, metadata, and embeddings
        """
        try:
            texts = [chunk["content"] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks using model: {self.model}")
            
            # Generate embeddings in batches to avoid API limits
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
            
            # Add embeddings to chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                chunk["embedding"] = embedding
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
