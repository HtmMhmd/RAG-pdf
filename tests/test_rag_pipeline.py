import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from app.config import Config
from app.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    def setup_method(self):
        # Create a temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test config
        self.config = Config()
        self.config.use_local_embeddings = True
        self.config.vector_db_path = os.path.join(self.temp_dir, "vector_db")
        
        # Create temporary test PDF path
        self.pdf_path = os.path.join(self.temp_dir, "test.pdf")
        with open(self.pdf_path, "w") as f:
            f.write("dummy pdf content")
            
        # Project ID for testing
        self.project_id = "test_project"
        
        # Mock components
        self.mock_pdf_processor = MagicMock()
        self.mock_embedding_generator = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_llm = MagicMock()
        
        # Initialize pipeline with mocks
        self.pipeline = RAGPipeline(self.config)
        self.pipeline.pdf_processor = self.mock_pdf_processor
        self.pipeline.embedding_generator = self.mock_embedding_generator
        self.pipeline.vector_db = self.mock_vector_db
        self.pipeline.llm = self.mock_llm
        
    def teardown_method(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_process_pdf_for_project(self):
        # Mock return values
        self.mock_pdf_processor.process_pdf.return_value = [
            {"content": "test content", "metadata": {"page": 1, "chunk_id": "1_0"}}
        ]
        self.mock_embedding_generator.generate_embeddings.return_value = [
            {"content": "test content", "metadata": {"page": 1, "chunk_id": "1_0"}, "embedding": [0.1] * 384}
        ]
        
        # Call the method
        self.pipeline.process_pdf_for_project(self.pdf_path, self.project_id)
        
        # Check if components were called correctly
        self.mock_pdf_processor.process_pdf.assert_called_once_with(self.pdf_path)
        self.mock_embedding_generator.generate_embeddings.assert_called_once()
        self.mock_vector_db.add_pdf_chunks_to_project.assert_called_once_with(
            self.mock_embedding_generator.generate_embeddings.return_value,
            self.pdf_path,
            self.project_id
        )
    
    def test_answer_question_for_project(self):
        # Mock context chunks return value
        mock_chunks = [
            {
                "content": "test content",
                "metadata": {"page": 1, "chunk_id": "1_0", "pdf_id": "test.pdf"}
            }
        ]
        self.mock_vector_db.query_project.return_value = mock_chunks
        
        # Mock answer generation
        mock_answer = "This is the answer"
        mock_citations = [{"pdf_id": "test.pdf", "page": 1}]
        self.mock_llm.generate_answer_with_pdf_citations.return_value = (mock_answer, mock_citations)
        
        # Call the method
        question = "test question"
        answer, citations = self.pipeline.answer_question_for_project(question, self.project_id)
        
        # Check if components were called correctly
        self.mock_vector_db.query_project.assert_called_once_with(question, self.project_id)
        self.mock_llm.generate_answer_with_pdf_citations.assert_called_once_with(question, mock_chunks)
        
        # Check return values
        assert answer == mock_answer
        assert citations == mock_citations
    
    def test_answer_question_for_pdf_in_project(self):
        # Mock context chunks return value
        mock_chunks = [
            {
                "content": "test content",
                "metadata": {"page": 1, "chunk_id": "1_0", "pdf_id": "test.pdf"}
            }
        ]
        self.mock_vector_db.query_pdf_in_project.return_value = mock_chunks
        
        # Mock answer generation
        mock_answer = "This is the answer"
        mock_citations = [1]  # Page numbers
        self.mock_llm.generate_answer.return_value = (mock_answer, mock_citations)
        
        # Call the method
        question = "test question"
        answer, citations = self.pipeline.answer_question_for_pdf_in_project(question, self.pdf_path, self.project_id)
        
        # Check if components were called correctly
        self.mock_vector_db.query_pdf_in_project.assert_called_once_with(question, self.pdf_path, self.project_id)
        self.mock_llm.generate_answer.assert_called_once_with(question, mock_chunks)
        
        # Check return values
        assert answer == mock_answer
        assert citations == mock_citations
