import logging
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
import re
import os
from langchain_text_splitters.nltk import NLTKTextSplitter
import nltk

# Download NLTK data on first import
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir='/usr/share/nltk_data')

# nltk.download('punkt_tab')

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, config):
        self.config = config
        # Initialize LangChain's NLTKTextSplitter
        self.text_splitter = NLTKTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            language="english"
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page numbers.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of tuples containing (text, page_number)
        """
        try:
            doc = fitz.open(pdf_path)
            text_with_pages = []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Clean text: remove excessive whitespace and normalize
                text = re.sub(r'\s+', ' ', text).strip()
                if text:  # Only add non-empty text
                    # 1-indexed page numbers
                    text_with_pages.append((text, page_num + 1))

            logger.info(f"Extracted text from {len(text_with_pages)} pages")
            return text_with_pages

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def chunk_text(self, text_with_pages: List[Tuple[str, int]]) -> List[dict]:
        """
        Split the extracted text into chunks while preserving page information.

        Args:
            text_with_pages: List of (text, page_number) tuples

        Returns:
            List of dictionaries with text chunks and their metadata
        """
        try:
            chunks = []

            for text, page_num in text_with_pages:
                # Split the text into chunks using LangChain's NLTKTextSplitter
                text_chunks = self.text_splitter.split_text(text)

                # Create chunk objects with metadata
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "page": page_num,
                            "chunk_id": f"page_{page_num}_chunk_{i}"
                        }
                    })

            logger.info(
                f"Created {len(chunks)} chunks from the extracted text using LangChain's NLTKTextSplitter")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> List[dict]:
        """
        Process a PDF file: extract text and split into chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with text chunks and their metadata
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            text_with_pages = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text_with_pages)
            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
