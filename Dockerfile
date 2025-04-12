FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model to ensure it's available for offline use
RUN python -c "from sentence_transformers import SentenceTransformer; \
               model = SentenceTransformer('all-MiniLM-L6-v2'); \
               print(f'Downloaded model with embedding size: {len(model.encode(\"test\"))}');"

# Set up DVC
RUN pip install dvc dvc-gdrive

# Copy application code
COPY . .

# Create directory for vector database
RUN mkdir -p data/vector_db

# Create directory for projects
RUN mkdir -p projects

# Initialize DVC if not already initialized
RUN if [ ! -d .dvc ]; then dvc init; fi

# Set environment variables
ENV USE_LOCAL_EMBEDDINGS=true
ENV LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PROJECTS_DIR=/app/projects

# Add DVC scripts
COPY scripts/dvc_helpers.sh /app/scripts/
RUN chmod +x /app/scripts/dvc_helpers.sh

# Default command
ENTRYPOINT ["python", "main.py"]
