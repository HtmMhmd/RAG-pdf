import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # OpenAI API configuration
        self.openai_api_key = self.get_env_var("OPENAI_API_KEY", "")
        self.embedding_model = self.get_env_var("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.llm_model = self.get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        
        # Local embedding model configuration
        self.use_local_embeddings = self.parse_bool(self.get_env_var("USE_LOCAL_EMBEDDINGS", "false"))
        self.local_embedding_model = self.get_env_var("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Vector DB configuration
        self.vector_db_path = self.get_env_var("VECTOR_DB_PATH", "data/vector_db")
        
        # Chunking configuration
        self.chunk_size = int(self.get_env_var("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(self.get_env_var("CHUNK_OVERLAP", "200"))
        
        # RAG configuration
        self.max_chunks = int(self.get_env_var("MAX_CHUNKS", "5"))
    
    def get_env_var(self, var_name, default=None):
        value = os.getenv(var_name, default)
        if value is None:
            if default is None:
                logger.error(f"Environment variable {var_name} not set and no default provided")
                raise ValueError(f"Environment variable {var_name} not set")
            logger.warning(f"Environment variable {var_name} not set, using default: {default}")
        return value
    
    def parse_bool(self, value):
        """Convert string value to boolean."""
        return str(value).lower() in ('true', 'yes', '1', 't', 'y')
