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
