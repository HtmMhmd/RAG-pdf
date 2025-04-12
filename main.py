import argparse
import logging
import os
import subprocess
import glob
from app.config import Config
from app.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_dvc_command(command):
    """Run a DVC command and log the output."""
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        logger.info(f"DVC command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e.stderr}")
        return False

def dvc_track_project(project_id):
    """Track the project's vector database with DVC."""
    return run_dvc_command(f"bash -c 'source scripts/dvc_helpers.sh && dvc_track_project {project_id}'")

def dvc_pull_project(project_id):
    """Pull the project's vector database from DVC remote."""
    return run_dvc_command(f"bash -c 'source scripts/dvc_helpers.sh && dvc_pull_project {project_id}'")

def find_pdf_files(path):
    """
    Find all PDF files in a directory or return the path if it's a single PDF.
    
    Args:
        path: Path to a PDF file or directory containing PDFs
        
    Returns:
        List of paths to PDF files
    """
    if os.path.isdir(path):
        # Find all PDFs in the directory and its subdirectories
        pdf_files = glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True)
        logger.info(f"Found {len(pdf_files)} PDF files in directory {path}")
        return pdf_files
    elif os.path.isfile(path) and path.lower().endswith('.pdf'):
        # Single PDF file
        return [path]
    else:
        logger.warning(f"Path {path} is not a PDF file or directory")
        return []

def main():
    parser = argparse.ArgumentParser(description="PDF RAG System")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file or directory containing PDFs")
    parser.add_argument("--project", type=str, help="Project ID to organize PDFs")
    parser.add_argument("--question", type=str, help="Question to ask about the PDF or project")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the vector index")
    parser.add_argument("--dvc-push", action="store_true", help="Push project vector DB to DVC remote after processing")
    parser.add_argument("--dvc-pull", action="store_true", help="Pull project vector DB from DVC remote before query")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set default project if not specified
    project_id = args.project if args.project else config.default_project
    
    # Pull from DVC if requested or if auto-pull is enabled
    if args.dvc_pull or config.dvc_auto_pull:
        logger.info(f"Pulling project {project_id} vector database from DVC remote...")
        dvc_pull_project(project_id)
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(config)
    
    if args.pdf:
        # Find all PDFs to process
        pdf_files = find_pdf_files(args.pdf)
        
        if not pdf_files:
            logger.error(f"No PDF files found at {args.pdf}")
            return
        
        # Process each PDF file found
        for pdf_path in pdf_files:
            logger.info(f"Processing PDF: {pdf_path} for project: {project_id}")
            pipeline.process_pdf_for_project(pdf_path, project_id)
        
        # Track with DVC if requested or if auto-push is enabled
        if args.dvc_push or config.dvc_auto_push:
            logger.info(f"Tracking project {project_id} vector database with DVC...")
            dvc_track_project(project_id)
        
        # If question provided and there's only one PDF, query that specific PDF
        if args.question and len(pdf_files) == 1:
            logger.info(f"Question about PDF {pdf_files[0]}: {args.question}")
            answer, citations = pipeline.answer_question_for_pdf_in_project(args.question, pdf_files[0], project_id)
            
            print("\nAnswer:")
            print(answer)
            
            if citations:
                print("\nCitations:")
                for citation in citations:
                    print(f"- Page {citation}")
        # If question provided but multiple PDFs processed, query the whole project
        elif args.question and len(pdf_files) > 1:
            logger.info(f"Question about project {project_id} with {len(pdf_files)} PDFs: {args.question}")
            answer, citations = pipeline.answer_question_for_project(args.question, project_id)
            
            print("\nAnswer:")
            print(answer)
            
            if citations:
                print("\nCitations:")
                for citation in citations:
                    print(f"- Document: {citation['pdf_id']}, Page {citation['page']}")
        else:
            logger.info(f"Processed {len(pdf_files)} PDFs and indexed in project {project_id}. Run again with a question to query.")
    
    elif args.question and project_id:
        # Query about an entire project
        logger.info(f"Question about project {project_id}: {args.question}")
        answer, citations = pipeline.answer_question_for_project(args.question, project_id)
        
        print("\nAnswer:")
        print(answer)
        
        if citations:
            print("\nCitations:")
            for citation in citations:
                print(f"- Document: {citation['pdf_id']}, Page {citation['page']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
