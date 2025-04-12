import logging
import os
from typing import Tuple, List, Dict, Any
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
