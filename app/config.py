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
        self.embedding_model = self.get_env_var(
            "EMBEDDING_MODEL", "text-embedding-ada-002")
        self.llm_model = self.get_env_var("LLM_MODEL", "gpt-3.5-turbo")

        # Local embedding model configuration
        self.use_local_embeddings = self.parse_bool(
            self.get_env_var("USE_LOCAL_EMBEDDINGS", "true"))
        self.local_embedding_model = self.get_env_var(
            "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # Project configuration
        self.default_project = self.get_env_var("DEFAULT_PROJECT", "default")
        self.projects_dir = self.get_env_var("PROJECTS_DIR", "projects")
        os.makedirs(self.projects_dir, exist_ok=True)

        # Vector DB configuration (legacy path, kept for backward compatibility)
        self.vector_db_path = self.get_env_var(
            "VECTOR_DB_PATH", "data/vector_db")

        # DVC configuration
        self.dvc_remote = self.get_env_var("DVC_REMOTE", "")
        self.dvc_auto_push = self.parse_bool(
            self.get_env_var("DVC_AUTO_PUSH", "false"))
        self.dvc_auto_pull = self.parse_bool(
            self.get_env_var("DVC_AUTO_PULL", "false"))

        # Chunking configuration
        # Updated default to 800 tokens
        self.chunk_size = int(self.get_env_var("CHUNK_SIZE", "800"))
        # Updated default to 200 tokens
        self.chunk_overlap = int(self.get_env_var("CHUNK_OVERLAP", "200"))

        # RAG configuration
        self.max_chunks = int(self.get_env_var("MAX_CHUNKS", "5"))

    def get_env_var(self, var_name, default=None):
        value = os.getenv(var_name, default)
        if value is None:
            if default is None:
                logger.error(
                    f"Environment variable {var_name} not set and no default provided")
                raise ValueError(f"Environment variable {var_name} not set")
            logger.warning(
                f"Environment variable {var_name} not set, using default: {default}")
        logger.warning(
            f"Environment variable {var_name} not set, using default: {value}")
        return value

    def parse_bool(self, value):
        """Convert string value to boolean."""
        return str(value).lower() in ('true', 'yes', '1', 't', 'y')

    def get_project_vector_db_path(self, project_id):
        """Get the path to the vector database for a specific project."""
        project_dir = os.path.join(self.projects_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)
        vector_db_path = os.path.join(project_dir, "vector_db")
        return vector_db_path
