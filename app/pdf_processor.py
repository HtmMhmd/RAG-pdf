import logging
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
import re
import nltk
from nltk.tokenize import word_tokenize

import os

# Use ./nltk_data in the workspace
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
# Create the directory if it doesn't exist
print(f"NLTK data directory: {nltk_data_dir}")
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_dir

# try:
#     # check the new location.
#     nltk.data.find('tokenizers/punkt/english.pickle')
# except LookupError:
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.data.find('tokenizers/punkt_tab/english/')


logger = logging.getLogger(__name__)


class NLTKTextSplitter:
    """
    Text splitter using NLTK for token-based chunking.
    """

    def __init__(self, chunk_size=800, chunk_overlap=200):
        """
        Initialize the NLTK text splitter.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        # Tokenize the text
        tokens = word_tokenize(text)

        if len(tokens) <= self.chunk_size:
            return [text]

        # Create chunks based on token count
        chunks = []
        start = 0
        while start < len(tokens):
            # Find end position for current chunk
            end = min(start + self.chunk_size, len(tokens))

            # Extract token sublist
            chunk_tokens = tokens[start:end]

            # Rejoin tokens to form a text chunk
            # We use the original text's spacing by finding the span in the original text
            chunk_text = ' '.join(chunk_tokens)

            chunks.append(chunk_text)

            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap if end < len(tokens) else end

        # Log chunk statistics
        logger.debug(
            f"Split text into {len(chunks)} chunks with NLTK tokenizer")

        return chunks


class PDFProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = NLTKTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
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
                # Split the text into chunks using NLTK tokenizer
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
                f"Created {len(chunks)} chunks from the extracted text using NLTK tokenizer")
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
