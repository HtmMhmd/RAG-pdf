import logging
import os
from typing import Tuple, List, Dict, Any, Optional
from .pdf_processor import PDFProcessor
from .embedding import EmbeddingGenerator
from .vector_db import VectorDB
from .llm import LLM

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_db = VectorDB(config)
        self.llm = LLM(config)
    
    def index_exists(self, pdf_path: str) -> bool:
        """Check if the vector index already exists for the given PDF."""
        return self.vector_db.collection_exists(pdf_path)
    
    def pdf_exists_in_project(self, pdf_path: str, project_id: str) -> bool:
        """Check if the PDF has been added to the project."""
        # Implement check logic
        pass
    
    def process_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF document and store its chunks in the vector database.
        
        Args:
            pdf_path: Path to the PDF file
        """
        try:
            # Extract and chunk text from PDF
            logger.info(f"Processing PDF: {pdf_path}")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # Generate embeddings for the chunks
            embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
            
            # Store chunks in the vector database
            self.vector_db.add_chunks_to_db(embedded_chunks, pdf_path)
            
            logger.info(f"Successfully processed and indexed PDF: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline: {e}")
            raise
    
    def process_pdf_for_project(self, pdf_path: str, project_id: str) -> None:
        """
        Process a PDF document and store its chunks in the project's vector database.
        
        Args:
            pdf_path: Path to the PDF file
            project_id: ID of the project this PDF belongs to
        """
        try:
            # Extract and chunk text from PDF
            logger.info(f"Processing PDF {pdf_path} for project {project_id}")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # Generate embeddings for the chunks
            embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
            
            # Store chunks in the vector database under the project
            self.vector_db.add_pdf_chunks_to_project(embedded_chunks, pdf_path, project_id)
            
            logger.info(f"Successfully processed and indexed PDF {pdf_path} for project {project_id}")
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline for project: {e}")
            raise
    
    def answer_question(self, question: str, pdf_path: str) -> Tuple[str, List[int]]:
        """
        Answer a question about the PDF document.
        
        Args:
            question: The user's question
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing the answer and a list of cited page numbers
        """
        try:
            # Ensure the PDF has been processed
            if not self.index_exists(pdf_path):
                logger.info(f"PDF {pdf_path} has not been processed yet, processing now...")
                self.process_pdf(pdf_path)
            
            # Retrieve relevant chunks from the vector database
            context_chunks = self.vector_db.query_db(question, pdf_path)
            
            # Generate answer using the LLM
            answer, citations = self.llm.generate_answer(question, context_chunks)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def answer_question_for_project(self, question: str, project_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Answer a question using all PDFs in a project.
        
        Args:
            question: The user's question
            project_id: ID of the project
            
        Returns:
            Tuple containing the answer and a list of citations with PDF info
        """
        try:
            # Retrieve relevant chunks from the project's vector database
            context_chunks = self.vector_db.query_project(question, project_id)
            
            if not context_chunks:
                return "No relevant information found in the project documents.", []
            
            # Generate answer using the LLM
            answer, citations = self.llm.generate_answer_with_pdf_citations(question, context_chunks)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Error answering question for project: {e}")
            raise
    
    def answer_question_for_pdf_in_project(self, question: str, pdf_path: str, project_id: str) -> Tuple[str, List[int]]:
        """
        Answer a question about a specific PDF in a project.
        
        Args:
            question: The user's question
            pdf_path: Path to the PDF file
            project_id: ID of the project
            
        Returns:
            Tuple containing the answer and a list of cited page numbers
        """
        try:
            # Retrieve relevant chunks for this PDF from the project's vector database
            context_chunks = self.vector_db.query_pdf_in_project(question, pdf_path, project_id)
            
            if not context_chunks:
                return "No relevant information found in this document.", []
            
            # Generate answer using the LLM
            answer, citations = self.llm.generate_answer(question, context_chunks)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Error answering question for PDF in project: {e}")
            raise
