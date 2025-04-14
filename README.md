# PDF RAG (Retrieval-Augmented Generation) System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com/)

A professional document question-answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate, contextual answers from PDF documents. This tool bridges the gap between unstructured PDF content and LLM capabilities, making your documents searchable and queryable through natural language.

## 📋 Key Features

- **PDF Text Extraction**: Efficiently extracts and processes text from PDF files
- **Intelligent Text Chunking**: Uses LangChain's NLTKTextSplitter for optimal retrieval
- **Flexible Embedding Options**: Choose between local embedding models or OpenAI API
- **Vector Database Storage**: Persistent ChromaDB integration for semantic search
- **Project-Based Organization**: Group related PDFs for cross-document querying
- **Version Control**: DVC integration for tracking and sharing vector databases
- **Docker Integration**: Containerization for consistent deployment and scaling
- **Devcontainer Support**: VS Code devcontainer for easy development setup

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git (for version control)
- OpenAI API key (optional, for OpenAI embeddings and LLM)

### Docker Installation (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-pdf.git
   cd rag-pdf
   ```

2. Create and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/htmmhmd/rag-pdf.git
   cd rag-pdf
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

## 💡 How to Use

### Quick Start Examples

**Process a PDF and ask a question:**
```bash
# With Docker
docker-compose run rag-pdf --pdf /app/pdfs/your_document.pdf --question "What are the key points in this document?"

# Without Docker
python main.py --pdf path/to/your/document.pdf --question "What are the key points in this document?"
```

**Process PDFs into a project:**
```bash
# Create a project and process multiple PDFs
docker-compose run rag-pdf --pdf /app/pdfs/document1.pdf --project "research_project"
docker-compose run rag-pdf --pdf /app/pdfs/document2.pdf --project "research_project"
```

**Query across a whole project:**
```bash
docker-compose run rag-pdf --project "research_project" --question "What are the common themes across these documents?"
```

### Using the Batch Processing Script

For processing multiple PDFs at once:

```bash
# Make the script executable
chmod +x scripts/process_project_pdfs.sh

# Process all PDFs in a directory
./scripts/process_project_pdfs.sh research_project
```

### API Usage Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  PDF Document   │────►│  Text Extraction │────►│  Text Chunking│
│                 │     │                  │     │               │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  User Query     │────►│  Vector Search   │◄────│  Vector DB    │
│                 │     │                  │     │               │
└─────────────────┘     └────────┬─────────┘     └───────▲───────┘
                                 │                       │
                                 ▼                       │
                        ┌──────────────────┐     ┌───────────────┐
                        │                  │     │               │
                        │  LLM Response    │     │  Embeddings   │
                        │                  │     │  Generation   │
                        └──────────────────┘     └───────────────┘
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_LOCAL_EMBEDDINGS` | Use local embedding models instead of OpenAI | `true` |
| `LOCAL_EMBEDDING_MODEL` | Local embedding model to use | `all-MiniLM-L6-v2` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-ada-002` |
| `LLM_MODEL` | OpenAI LLM model | `gpt-4o-mini` |
| `DEFAULT_PROJECT` | Default project name | `default` |
| `VECTOR_DB_PATH` | Path to vector database | `/app/data/vector_db` |
| `DVC_REMOTE` | URL of DVC remote | none |
| `DVC_AUTO_PUSH` | Push to DVC after processing | `false` |
| `DVC_AUTO_PULL` | Pull from DVC before querying | `false` |

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--pdf` | Path to PDF file or directory of PDFs |
| `--question` | Question to ask about the document(s) |
| `--project` | Project name to organize related PDFs |
| `--rebuild-index` | Force rebuild of the vector index |
| `--dvc-push` | Push vector DB to DVC remote after processing |
| `--dvc-pull` | Pull vector DB from DVC remote before querying |

## 🔍 Detailed System Explanation

### RAG Pipeline Architecture

The PDF RAG system implements a Retrieval-Augmented Generation pipeline to enhance LLM responses with relevant context from PDF documents:

1. **Document Processing Layer**
   - PDF text extraction using PyMuPDF
   - Text normalization and cleaning
   - Intelligent chunking with LangChain's NLTKTextSplitter
   - Document metadata preservation (page numbers, chunk IDs)

2. **Vector Representation Layer**
   - Embedding generation with local models (SentenceTransformers) or OpenAI
   - Vector storage in ChromaDB for efficient similarity search
   - Project-based organization for document collections
   - Version control with DVC for sharing and collaboration

3. **Retrieval Layer**
   - Semantic search for relevant document chunks
   - Metadata-based filtering capabilities
   - Configurable retrieval parameters (chunk count, similarity threshold)
   - Cross-document context retrieval within projects

4. **Generation Layer**
   - Context-enhanced prompts for LLMs
   - Page citation integration in responses
   - Fallback mechanisms for API failures
   - Response formatting and post-processing

### Key Components

#### PDF Processor

The `PDFProcessor` class handles the extraction and chunking of PDF text:

```python
def process_pdf(self, pdf_path: str) -> List[dict]:
    """
    Process a PDF file: extract text and split into chunks.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries with text chunks and their metadata
    """
    logger.info(f"Processing PDF: {pdf_path}")
    text_with_pages = self.extract_text_from_pdf(pdf_path)
    chunks = self.chunk_text(text_with_pages)
    return chunks
```

#### Vector Database

The system uses ChromaDB as its vector store, with methods for adding and retrieving chunks:

```python
def add_chunks_to_db(self, chunks: List[Dict[str, Any]], pdf_path: str) -> None:
    # Create or replace collection
    if self.collection_exists(pdf_path):
        collection_name = self.get_collection_name(pdf_path)
        self.client.delete_collection(collection_name)
    
    collection = self.create_collection(pdf_path)
    
    # Prepare data for insertion
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Insert with embeddings if available
    if "embedding" in chunks[0] and chunks[0]["embedding"] is not None:
        embeddings = [chunk["embedding"] for chunk in chunks]
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

#### DVC Integration

DVC is used for version controlling the vector databases:

```python
def push_to_dvc(self, project_name: str) -> bool:
    """Push project vector DB to DVC remote"""
    vector_db_path = os.path.join(self.projects_dir, project_name, "vector_db")
    
    if not os.path.exists(vector_db_path):
        logger.error(f"Vector DB not found at {vector_db_path}")
        return False
    
    # Add to DVC
    try:
        subprocess.run(["dvc", "add", vector_db_path], check=True)
        subprocess.run(["dvc", "push", f"{vector_db_path}.dvc"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC error: {e}")
        return False
```

## DVC for Vector Database Version Control

### Using DVC with the Research Project

The repository includes a pre-configured research project with its vector database already tracked in DVC. To use it:

1. **Pull the research project vector database:**
   ```bash
   # Navigate to the project directory
   cd /workspaces/RAG-pdf
   
   # Ensure the DVC files aren't git-ignored
   # If you see an error about "bad DVC file name is git-ignored", check your .gitignore file
   # Make sure it contains these lines to ignore the data but not the DVC tracking files:
   # projects/research_project/vector_db/**
   # !projects/research_project/vector_db.dvc
   
   # Pull the specific research project vector database
   dvc pull projects/research_project/vector_db.dvc
   ```

2. **Verify the vector database was pulled correctly:**
   ```bash
   # Check that the vector database files exist
   ls -la projects/research_project/vector_db
   ```

3. **Ask questions using the pre-built vector database:**
   ```bash
   # Use Docker Compose
   docker-compose run rag-pdf --project "research_project" --question "What are the key findings in these research papers?"
   
   # Or with Python directly
   python main.py --project "research_project" --question "What are the key findings in these research papers?"
   ```

4. **Update the vector database after adding new PDFs:**
   ```bash
   # Process new PDFs
   docker-compose run rag-pdf --pdf /app/pdfs/new_paper.pdf --project "research_project"
   
   # Update the DVC tracking
   dvc add projects/research_project/vector_db
   git add projects/research_project/vector_db.dvc
   git commit -m "Update research project vector database"
   dvc push projects/research_project/vector_db.dvc
   ```

### Sharing the Research Project

To share the research project with team members:

1. **Push your changes to the repository:**
   ```bash
   git push origin main
   ```

2. **Team members can then pull both the code and the vector database:**
   ```bash
   git pull
   dvc pull projects/research_project/vector_db.dvc
   ```

3. **Now they can immediately query the same document collection:**
   ```bash
   docker-compose run rag-pdf --project "research_project" --question "Summarize the main topics in these papers"
   ```

## 📊 Usage Examples

### Example 1: Processing a Technical PDF and Asking Questions

```bash
# Process a technical manual
docker-compose run rag-pdf --pdf /app/pdfs/technical_manual.pdf --project "manuals"

# Ask specific questions about the technical content
docker-compose run rag-pdf --project "manuals" --question "How do I configure the network settings according to the manual?"
```

Expected output:
```
According to the technical manual, you can configure network settings by:
1. Navigating to Settings > Network on the main control panel
2. Selecting either DHCP for automatic configuration or Static IP for manual setup
3. For Static IP, entering the required fields: IP Address, Subnet Mask, Gateway, and DNS servers
4. Clicking Apply to save the changes
5. Restarting the network service by clicking on "Restart Network"

[Page 24] The manual notes that you should record your settings before making changes in case you need to revert.
```

### Example 2: Research Paper Analysis

```bash
# Process multiple research papers into a project
./scripts/process_project_pdfs.sh research_project projects/research_project

# Ask for a synthesis of findings across papers
docker-compose run rag-pdf --project "research_project" --question "What are the common challenges in implementing transformer models according to these papers?"
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Bug Reports and Feature Requests**
   - Open an issue on GitHub describing the bug or feature
   - Include steps to reproduce for bugs
   - Explain the value and use case for features

2. **Code Contributions**
   - Fork the repository
   - Create a feature branch: `git checkout -b feature/amazing-feature`
   - Commit your changes: `git commit -m 'Add some amazing feature'`
   - Push to the branch: `git push origin feature/amazing-feature`
   - Open a pull request

3. **Documentation**
   - Help improve the documentation with corrections or additions
   - Add examples to showcase different use cases

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing and LLM integration tools
- [ChromaDB](https://github.com/chroma-core/chroma) for the vector database
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF text extraction
- [NLTK](https://www.nltk.org/) for text chunking capabilities
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) for embedding models
- [DVC](https://dvc.org/) for vector database version control
- [Docker](https://www.docker.com/) for containerization support

## 📬 Contact

For questions, feedback, or support, please:
- Open an issue on GitHub
- Contact the maintainer at [your-email@example.com](mailto:your-email@example.com)
- Join our [Discord community](https://discord.gg/example)

---

**Note**: This README provides comprehensive documentation for the PDF RAG system. For further details, refer to the inline code documentation and comments.
