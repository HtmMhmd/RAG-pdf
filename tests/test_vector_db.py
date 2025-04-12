import os
import shutil
import tempfile
import pytest
import numpy as np
from app.config import Config
from app.vector_db import VectorDB

class TestVectorDB:
    def setup_method(self):
        # Create a temporary directory for vector db
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test config with the temporary directory
        self.config = Config()
        self.config.vector_db_path = self.temp_dir
        self.config.use_local_embeddings = True  # Use local embeddings for tests
        
        # Initialize VectorDB
        self.vector_db = VectorDB(self.config)
        
        # Test data
        self.project_id = "test_project"
        self.pdf_path = "/path/to/test.pdf"
        self.pdf_path2 = "/path/to/test2.pdf"
        
        # Sample chunks with embeddings
        self.chunks = [
            {
                "content": "This is test content 1.",
                "metadata": {"page": 1, "chunk_id": "page_1_chunk_0"},
                "embedding": [0.1] * 384  # Simplified embedding vector
            },
            {
                "content": "This is test content 2.",
                "metadata": {"page": 2, "chunk_id": "page_2_chunk_0"},
                "embedding": [0.2] * 384
            }
        ]
        
    def teardown_method(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_project_collection_name(self):
        collection_name = self.vector_db.get_project_collection_name(self.project_id)
        assert collection_name.startswith("project_")
        assert len(collection_name) > 8  # Should be a meaningful name
    
    def test_project_collection_exists(self):
        # Initially should not exist
        assert not self.vector_db.project_collection_exists(self.project_id)
        
        # Create the collection
        self.vector_db.create_project_collection(self.project_id)
        
        # Now should exist
        assert self.vector_db.project_collection_exists(self.project_id)
    
    def test_add_pdf_chunks_to_project(self):
        # Add chunks to project
        self.vector_db.add_pdf_chunks_to_project(self.chunks, self.pdf_path, self.project_id)
        
        # Check that collection exists
        assert self.vector_db.project_collection_exists(self.project_id)
        
        # Get collection and check contents
        collection = self.vector_db.get_project_collection(self.project_id)
        
        # Query to check if contents were added
        results = collection.get()
        
        # Should have 2 items
        assert len(results["ids"]) == 2
        
        # PDF ID should be in the metadata
        for metadata in results["metadatas"]:
            assert "pdf_id" in metadata
            assert metadata["pdf_id"] == os.path.basename(self.pdf_path)
    
    def test_remove_pdf_from_project(self):
        # Add chunks to project
        self.vector_db.add_pdf_chunks_to_project(self.chunks, self.pdf_path, self.project_id)
        
        # Add chunks from a second PDF
        chunks2 = self.chunks.copy()
        for chunk in chunks2:
            chunk["metadata"] = chunk["metadata"].copy()  # Make a copy of metadata
        
        self.vector_db.add_pdf_chunks_to_project(chunks2, self.pdf_path2, self.project_id)
        
        # Check initial state
        collection = self.vector_db.get_project_collection(self.project_id)
        results = collection.get()
        assert len(results["ids"]) == 4  # 2 chunks from each PDF
        
        # Remove the first PDF
        self.vector_db.remove_pdf_from_project(self.pdf_path, self.project_id)
        
        # Check that only second PDF remains
        results = collection.get()
        assert len(results["ids"]) == 2
        
        # All remaining items should be from the second PDF
        for metadata in results["metadatas"]:
            assert metadata["pdf_id"] == os.path.basename(self.pdf_path2)
    
    def test_query_project(self):
        # Add chunks to project
        self.vector_db.add_pdf_chunks_to_project(self.chunks, self.pdf_path, self.project_id)
        
        # Query the project
        results = self.vector_db.query_project("test content", self.project_id)
        
        # Should return chunks
        assert len(results) > 0
        
        # Check structure of returned chunks
        for chunk in results:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "page" in chunk["metadata"]
            assert "pdf_id" in chunk["metadata"]
    
    def test_query_pdf_in_project(self):
        # Add chunks from two PDFs to the project
        self.vector_db.add_pdf_chunks_to_project(self.chunks, self.pdf_path, self.project_id)
        
        chunks2 = [
            {
                "content": "This is other test content.",
                "metadata": {"page": 1, "chunk_id": "page_1_chunk_0"},
                "embedding": [0.3] * 384
            }
        ]
        self.vector_db.add_pdf_chunks_to_project(chunks2, self.pdf_path2, self.project_id)
        
        # Query specifically for chunks from pdf_path
        results = self.vector_db.query_pdf_in_project("test content", self.pdf_path, self.project_id)
        
        # Should return chunks only from the specified PDF
        assert len(results) > 0
        for chunk in results:
            assert chunk["metadata"]["pdf_id"] == os.path.basename(self.pdf_path)
