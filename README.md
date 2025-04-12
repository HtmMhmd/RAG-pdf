# PDF RAG (Retrieval-Augmented Generation) System

A professional system for answering questions about PDF documents using RAG (Retrieval-Augmented Generation) with LLMs.

## Features

- Extract text from PDF documents
- Process and chunk text for optimal retrieval
- Generate embeddings for semantic search
- Store and retrieve vectors using ChromaDB
- Answer questions using LLMs with context from the PDF
- Docker containerization for easy deployment
- Devcontainer setup for consistent development environment

## Requirements

- Python 3.9+
- Docker (for containerized deployment)
- OpenAI API key (or another LLM provider)

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-pdf.git
cd rag-pdf
```

2. Copy the example environment file and edit it with your settings:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. Run with Docker:
```bash
docker build -t rag-pdf .
docker run -it --env-file .env -v $(pwd)/sample_pdfs:/app/sample_pdfs rag-pdf --pdf sample_pdfs/your_document.pdf --question "Your question about the document?"
```

## Development Setup

### Using VS Code Devcontainer

1. Ensure you have Docker and VS Code with the Remote - Containers extension installed.
2. Open the project folder in VS Code.
3. Click on "Reopen in Container" when prompted, or run the "Remote-Containers: Reopen in Container" command from the Command Palette.
4. The container will build and set up the development environment automatically.

### Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Process a PDF and ask a question:

```bash
python main.py --pdf path/to/your/document.pdf --question "Your question about the document?"
```

Process a PDF without asking a question (to build the index):

```bash
python main.py --pdf path/to/your/document.pdf
```

Force rebuild of the index:

```bash
python main.py --pdf path/to/your/document.pdf --rebuild-index
```

## How It Works: RAG Pipeline Explained

This system uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about PDF documents. Here's a step-by-step explanation of how it works:

### 1. PDF Processing and Text Extraction

First, the system extracts and processes text from the PDF document:

```python
# Extract text from PDF with page numbers
def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
    doc = fitz.open(pdf_path)
    text_with_pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # Clean text: remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        if text:  # Only add non-empty text
            text_with_pages.append((text, page_num + 1))  # 1-indexed page numbers
            
    return text_with_pages
```

The extracted text is then split into smaller chunks for efficient processing:

```python
# Split text into chunks
def chunk_text(self, text_with_pages: List[Tuple[str, int]]) -> List[dict]:
    chunks = []
    
    for text, page_num in text_with_pages:
        # Split the text into chunks
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
            
    return chunks
```

### 2. Embedding Generation

Next, the system generates embeddings for each text chunk. The system supports both OpenAI API embeddings and local embedding models:

```python
def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [chunk["content"] for chunk in chunks]
    
    # First check if we're configured to use local embeddings by default
    if self.use_local_embeddings:
        all_embeddings = self.generate_embeddings_with_local_model(texts)
    else:
        # Try with OpenAI first, fallback to local model if it fails
        try:
            all_embeddings = self.generate_embeddings_with_openai(texts)
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}. Falling back to local embedding model.")
            all_embeddings = self.generate_embeddings_with_local_model(texts)
    
    # Add embeddings to chunks
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        chunk["embedding"] = embedding
    
    return chunks
```

#### Local Embedding Generation

```python
def generate_embeddings_with_local_model(self, texts: List[str]) -> List[List[float]]:
    if self.local_model is None:
        self.local_model = SentenceTransformer(self.local_model_name)
    
    # Generate embeddings in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = self.local_model.encode(batch_texts).tolist()
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings
```

### 3. Vector Database Storage

The system stores text chunks and their embeddings in a vector database (ChromaDB):

```python
def add_chunks_to_db(self, chunks: List[Dict[str, Any]], pdf_path: str) -> None:
    # Create a new collection or get existing one
    if self.collection_exists(pdf_path):
        collection_name = self.get_collection_name(pdf_path)
        self.client.delete_collection(collection_name)
    
    collection = self.create_collection(pdf_path)
    
    # Prepare data for insertion
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Check if chunks already have embeddings
    if "embedding" in chunks[0] and chunks[0]["embedding"] is not None:
        embeddings = [chunk["embedding"] for chunk in chunks]
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    else:
        # If no pre-generated embeddings, let ChromaDB handle embedding
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
```

### 4. Question Answering

When a question is asked, the system:

1. Retrieves relevant chunks from the vector database
2. Passes these chunks and the question to an LLM
3. Returns the generated answer with citations

#### Vector Search

```python
def query_db(self, question: str, pdf_path: str, n_results: int = None) -> List[Dict[str, Any]]:
    if n_results is None:
        n_results = self.config.max_chunks
        
    try:
        collection = self.get_collection(pdf_path)
        
        # Try standard query first, fallback to local embeddings if it fails
        try:
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
        except Exception:
            # Generate embedding using our local model
            query_embedding = self.get_query_embedding(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        
        # Convert results to a convenient format
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return chunks
    except Exception:
        # Fallback to empty results if all else fails
        return []
```

#### Answer Generation

```python
def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
    # Build the prompt
    prompt = self.build_prompt(question, context_chunks)
    
    try:
        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document excerpts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback answer generation without LLM
        context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])
        answer = (
            f"I found the following information in the document that may help answer your question about '{question}':\n\n"
            f"{context_text}\n\n"
            f"The above excerpts are from the document and may contain relevant information to answer your question."
        )
    
    # Extract citation page numbers
    citations = list(set(re.findall(r'\[Page\s+(\d+)\]', answer)))
    citations = [int(page) for page in citations]
    
    return answer, citations
```

### 5. Overall Pipeline Pseudocode

Here's a pseudocode representation of the entire RAG pipeline:

```python
# Pseudocode for the RAG pipeline
def process_pdf_and_answer_question(pdf_path, question):
    # Step 1: Extract and chunk text
    text_with_pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text_with_pages)
    
    # Step 2: Generate embeddings
    chunks_with_embeddings = generate_embeddings(chunks)
    
    # Step 3: Store in vector database
    add_chunks_to_db(chunks_with_embeddings, pdf_path)
    
    # Step 4: Query database and generate answer
    context_chunks = query_db(question, pdf_path)
    answer, citations = generate_answer(question, context_chunks)
    
    return answer, citations
```

## Deployment

The system can be deployed using Docker to various platforms:

### AWS ECS

1. Build and push the Docker image to ECR:
```bash
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com
docker build -t your-repo/rag-pdf .
docker tag your-repo/rag-pdf your-account-id.dkr.ecr.your-region.amazonaws.com/your-repo/rag-pdf
docker push your-account-id.dkr.ecr.your-region.amazonaws.com/your-repo/rag-pdf
```

2. Create an ECS task definition that uses the image and includes your environment variables.
3. Create an ECS service to run the task.

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/your-project/rag-pdf
gcloud run deploy rag-pdf --image gcr.io/your-project/rag-pdf --platform managed
```

## CI/CD with GitHub Actions

Here's a basic GitHub Actions workflow to build and deploy the Docker image:

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v1
      
      - name: Login to Container Registry
        uses: docker/login-action@v1
        with:
          registry: your-registry.com
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v2
        with:
          context: .
          push: true
          tags: your-registry.com/your-repo/rag-pdf:latest
      
      - name: Deploy to your platform
        run: |
          # Add deployment commands for your platform
          # e.g., kubectl apply, aws ecs update-service, etc.
```

## License

[MIT License](LICENSE)
