import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, config):
        self.config = config
        self.db_path = config.vector_db_path
        
        # Create the base directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize local model for direct embedding when needed
        self.local_model = None
        if config.use_local_embeddings:
            self.local_model = SentenceTransformer(config.local_embedding_model)
        
        # Current project client cache
        self._project_clients = {}
        self.default_embedding_function = self.get_embedding_function(config)
    
    def get_embedding_function(self, config):
        """Get the appropriate embedding function based on config."""
        if config.use_local_embeddings:
            logger.info(f"Using local embedding function with model: {config.local_embedding_model}")
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.local_embedding_model)
        elif "ada" in config.embedding_model or "openai" in config.embedding_model.lower():
            # Only use OpenAI if we have a valid API key
            if config.openai_api_key and config.openai_api_key.strip():
                logger.info(f"Using OpenAI embedding function with model: {config.embedding_model}")
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.openai_api_key,
                    model_name=config.embedding_model
                )
        
        # Default to a local embedding model
        logger.info(f"Using default local embedding function")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.local_embedding_model)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate an embedding for a query using the local model."""
        if self.local_model is None:
            self.local_model = SentenceTransformer(self.config.local_embedding_model)
        
        logger.info(f"Generating query embedding using local model")
        return self.local_model.encode(query).tolist()
    
    def get_project_db_path(self, project_id: str) -> str:
        """Get the database path for a specific project."""
        project_dir = os.path.join(self.config.projects_dir, project_id)
        vector_db_path = os.path.join(project_dir, "vector_db")
        os.makedirs(vector_db_path, exist_ok=True)
        return vector_db_path
    
    def get_project_client(self, project_id: str) -> chromadb.PersistentClient:
        """Get a ChromaDB client for a specific project."""
        if project_id not in self._project_clients:
            db_path = self.get_project_db_path(project_id)
            self._project_clients[project_id] = chromadb.PersistentClient(path=db_path)
        return self._project_clients[project_id]
    
    def get_project_collection_name(self, project_id: str) -> str:
        """Generate a unique collection name for the project."""
        return f"project_{hashlib.md5(project_id.encode()).hexdigest()}"
    
    def project_collection_exists(self, project_id: str) -> bool:
        """Check if a collection for the given project already exists."""
        collection_name = self.get_project_collection_name(project_id)
        try:
            client = self.get_project_client(project_id)
            collections = client.list_collections()
            return any(collection.name == collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Error checking project collection existence: {e}")
            return False
    
    def create_project_collection(self, project_id: str) -> chromadb.Collection:
        """Create a new collection for the project."""
        collection_name = self.get_project_collection_name(project_id)
        try:
            client = self.get_project_client(project_id)
            return client.create_collection(
                name=collection_name,
                embedding_function=self.default_embedding_function
            )
        except Exception as e:
            logger.error(f"Error creating project collection: {e}")
            raise
    
    def get_project_collection(self, project_id: str) -> chromadb.Collection:
        """Get the collection for the project."""
        collection_name = self.get_project_collection_name(project_id)
        try:
            client = self.get_project_client(project_id)
            return client.get_collection(
                name=collection_name,
                embedding_function=self.default_embedding_function
            )
        except Exception as e:
            logger.error(f"Error getting project collection: {e}")
            raise
    
    def add_pdf_chunks_to_project(self, chunks: List[Dict[str, Any]], pdf_path: str, project_id: str) -> None:
        """
        Add text chunks from a PDF to a project collection.
        
        Args:
            chunks: List of dictionaries with text chunks and metadata
            pdf_path: Path to the PDF file
            project_id: ID of the project this PDF belongs to
        """
        try:
            # Prepare PDF identifier
            pdf_id = os.path.basename(pdf_path)
            
            # Create a new collection or get existing one
            if not self.project_collection_exists(project_id):
                collection = self.create_project_collection(project_id)
                logger.info(f"Created new collection for project: {project_id}")
            else:
                collection = self.get_project_collection(project_id)
                logger.info(f"Using existing collection for project: {project_id}")
            
            # Update metadata to include PDF ID information
            for chunk in chunks:
                chunk["metadata"]["pdf_id"] = pdf_id
                # Create unique chunk ID for this PDF in the project
                page = chunk["metadata"]["page"]
                old_chunk_id = chunk["metadata"]["chunk_id"]
                chunk["metadata"]["chunk_id"] = f"{pdf_id}_{old_chunk_id}"
            
            # Prepare data for insertion
            ids = [f"{project_id}_{chunk['metadata']['pdf_id']}_{chunk['metadata']['chunk_id']}" for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Check if chunks already have embeddings from the embedding generator
            if "embedding" in chunks[0] and chunks[0]["embedding"] is not None:
                embeddings = [chunk["embedding"] for chunk in chunks]
                logger.info(f"Using pre-generated embeddings for vector database")
                
                # Add documents directly with embeddings to avoid re-embedding
                try:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                except TypeError:
                    # Some versions of ChromaDB might have different signature
                    # Try adding chunks one by one with embeddings
                    logger.warning("Falling back to individual chunk insertion")
                    for i in range(len(ids)):
                        collection.add(
                            ids=[ids[i]],
                            documents=[documents[i]],
                            metadatas=[metadatas[i]],
                            embeddings=[embeddings[i]]
                        )
            else:
                # If no pre-generated embeddings, let ChromaDB handle embedding
                logger.info("No pre-generated embeddings found, letting ChromaDB handle embedding")
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(chunks)} chunks from PDF {pdf_id} to project {project_id}")
            
        except Exception as e:
            logger.error(f"Error adding PDF chunks to project: {e}")
            # If we get embedding errors, try again with just the text
            if "embedding" in str(e).lower() or "openai" in str(e).lower():
                logger.info("Retrying without embeddings and letting ChromaDB use local model")
                try:
                    # Force using a local embedding model for this collection
                    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.config.local_embedding_model
                    )
                    
                    # Get or create the collection with local embedding function
                    collection_name = self.get_project_collection_name(project_id)
                    if self.project_collection_exists(project_id):
                        # Get existing collection with local embeddings
                        collection = self.get_project_client(project_id).get_collection(
                            name=collection_name,
                            embedding_function=local_ef
                        )
                    else:
                        # Create new collection with local embeddings
                        collection = self.get_project_client(project_id).create_collection(
                            name=collection_name,
                            embedding_function=local_ef
                        )
                    
                    # Add documents without providing embeddings
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info("Successfully added documents with local embedding function")
                    return
                except Exception as inner_e:
                    logger.error(f"Error in fallback embedding strategy: {inner_e}")
            raise
    
    def remove_pdf_from_project(self, pdf_path: str, project_id: str) -> None:
        """
        Remove all chunks for a specific PDF from a project collection.
        
        Args:
            pdf_path: Path to the PDF file
            project_id: ID of the project
        """
        try:
            if not self.project_collection_exists(project_id):
                logger.warning(f"Project collection {project_id} does not exist")
                return
                
            collection = self.get_project_collection(project_id)
            pdf_id = os.path.basename(pdf_path)
            
            # Query for all chunks belonging to this PDF
            results = collection.get(
                where={"pdf_id": pdf_id}
            )
            
            if results and results["ids"]:
                # Delete all chunks for this PDF
                collection.delete(
                    ids=results["ids"]
                )
                logger.info(f"Removed {len(results['ids'])} chunks for PDF {pdf_id} from project {project_id}")
            else:
                logger.info(f"No chunks found for PDF {pdf_id} in project {project_id}")
                
        except Exception as e:
            logger.error(f"Error removing PDF from project: {e}")
            raise
    
    def query_project(self, question: str, project_id: str, n_results: int = None) -> List[Dict[str, Any]]:
        """
        Query the vector database to find chunks relevant to the question within a project.
        
        Args:
            question: The user's question
            project_id: ID of the project to query
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with their content and metadata
        """
        if n_results is None:
            n_results = self.config.max_chunks
            
        try:
            if not self.project_collection_exists(project_id):
                logger.warning(f"Project collection {project_id} does not exist")
                return []
                
            collection = self.get_project_collection(project_id)
            
            # Try the standard query first
            try:
                results = collection.query(
                    query_texts=[question],
                    n_results=n_results
                )
            except Exception as e:
                # If we get an error, use local embedding instead
                logger.warning(f"Error using collection.query: {e}. Falling back to query_by_embeddings.")
                
                # Generate embedding using our local model
                query_embedding = self.get_query_embedding(question)
                
                # Query using the embedding directly
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            
            # Convert results to a more convenient format
            chunks = []
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for the query in project {project_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error querying project database: {e}")
            
            # Use fallback methods similar to the original method
            return []
    
    def query_pdf_in_project(self, question: str, pdf_path: str, project_id: str, n_results: int = None) -> List[Dict[str, Any]]:
        """
        Query for chunks from a specific PDF in a project.
        
        Args:
            question: The user's question
            pdf_path: Path to the PDF file
            project_id: ID of the project
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with their content and metadata
        """
        if n_results is None:
            n_results = self.config.max_chunks
            
        try:
            if not self.project_collection_exists(project_id):
                logger.warning(f"Project collection {project_id} does not exist")
                return []
                
            collection = self.get_project_collection(project_id)
            pdf_id = os.path.basename(pdf_path)
            
            # Try the standard query with where filter
            try:
                results = collection.query(
                    query_texts=[question],
                    n_results=n_results,
                    where={"pdf_id": pdf_id}
                )
            except Exception as e:
                # If we get an error, use local embedding instead
                logger.warning(f"Error using collection.query: {e}. Falling back to query_by_embeddings.")
                
                # Generate embedding using our local model
                query_embedding = self.get_query_embedding(question)
                
                # Query using the embedding directly
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where={"pdf_id": pdf_id}
                )
            
            # Convert results to a more convenient format
            chunks = []
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for PDF {pdf_id} in project {project_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error querying PDF in project: {e}")
            return []
    
    def pdf_exists_in_project(self, pdf_path: str, project_id: str) -> bool:
        """Check if a PDF has already been indexed in the project.
        
        Args:
            pdf_path: Path to the PDF file
            project_id: ID of the project
            
        Returns:
            True if the PDF has already been indexed, False otherwise
        """
        try:
            if not self.project_collection_exists(project_id):
                return False
                
            collection = self.get_project_collection(project_id)
            pdf_id = os.path.basename(pdf_path)
            
            # Try to find at least one chunk for this PDF
            results = collection.get(
                limit=1,
                where={"pdf_id": pdf_id}
            )
            
            # If we found at least one chunk, the PDF is already indexed
            return len(results["ids"]) > 0
            
        except Exception as e:
            logger.error(f"Error checking if PDF exists in project: {e}")
            # If there's an error, assume PDF is not indexed to be safe
            return False
