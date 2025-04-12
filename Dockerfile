FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model to ensure it's available for offline use
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create directory for vector database
RUN mkdir -p data/vector_db

# Set environment variables to use the local model by default
ENV USE_LOCAL_EMBEDDINGS=true
ENV LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Default command
ENTRYPOINT ["python", "main.py"]
