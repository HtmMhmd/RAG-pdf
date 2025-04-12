import pytest
import os
from app.config import Config
from app.embedding import EmbeddingGenerator

class TestEmbedding:
    def setup_method(self):
        # Create test config with local embeddings enabled
        self.config = Config()
        self.config.use_local_embeddings = True
        self.config.local_embedding_model = "all-MiniLM-L6-v2"
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(self.config)
        
        # Test data
        self.chunks = [
            {
                "content": "This is test content 1.",
                "metadata": {"page": 1, "chunk_id": "page_1_chunk_0"}
            },
            {
                "content": "This is test content 2.",
                "metadata": {"page": 2, "chunk_id": "page_2_chunk_0"}
            }
        ]
        
    def test_local_embedding_generation(self):
        # Generate embeddings using local model
        embedded_chunks = self.embedding_generator.generate_embeddings(self.chunks)
        
        # Check if embeddings were added
        assert len(embedded_chunks) == 2
        for chunk in embedded_chunks:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            # Local embeddings are typically 384-dimensional for all-MiniLM-L6-v2
            assert len(chunk["embedding"]) > 0
    
    def test_local_model_initialization(self):
        # Test that the model is initialized when needed
        assert self.embedding_generator.local_model is not None
        
        # Reset model
        self.embedding_generator.local_model = None
        assert self.embedding_generator.local_model is None
        
        # Embedding generation should initialize model
        self.embedding_generator.generate_embeddings_with_local_model(["Test content"])
        assert self.embedding_generator.local_model is not None
    
    def test_fallback_mechanism(self):
        # Set up config to try OpenAI first but allow fallback
        self.config.use_local_embeddings = False
        self.config.openai_api_key = "invalid_key_to_force_fallback"
        
        # Create new embedding generator with this config
        embedding_gen = EmbeddingGenerator(self.config)
        
        # Generate embeddings - should fall back to local model
        embedded_chunks = embedding_gen.generate_embeddings(self.chunks)
        
        # Check if embeddings were generated despite OpenAI API failure
        assert len(embedded_chunks) == 2
        for chunk in embedded_chunks:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) > 0
