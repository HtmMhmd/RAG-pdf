import os
import pytest
from app.config import Config
from app.pdf_processor import PDFProcessor

class TestPDFProcessor:
    def setup_method(self):
        self.config = Config()
        self.pdf_processor = PDFProcessor(self.config)
        
        # Create a test directory if it doesn't exist
        os.makedirs("tests/test_data", exist_ok=True)
    
    def test_chunk_text(self):
        # Test with a simple text and page number
        text_with_pages = [
            ("This is a test document. It contains multiple sentences that should be chunked appropriately.", 1),
            ("This is page 2. It also has content that will be chunked.", 2)
        ]
        
        chunks = self.pdf_processor.chunk_text(text_with_pages)
        
        # Check that chunks were created
        assert len(chunks) > 0
        
        # Check that each chunk has the expected structure
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "page" in chunk["metadata"]
            assert "chunk_id" in chunk["metadata"]
            
            # Page should be either 1 or 2
            assert chunk["metadata"]["page"] in [1, 2]
            
            # Content should be a non-empty string
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0
