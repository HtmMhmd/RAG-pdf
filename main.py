import argparse
import logging
import os
from app.config import Config
from app.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="PDF RAG System")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--project", type=str, help="Project ID to organize PDFs")
    parser.add_argument("--question", type=str, help="Question to ask about the PDF or project")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the vector index")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set default project if not specified
    project_id = args.project if args.project else config.default_project
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(config)
    
    if args.pdf:
        # Process a specific PDF for a project
        logger.info(f"Processing PDF: {args.pdf} for project: {project_id}")
        pipeline.process_pdf_for_project(args.pdf, project_id)
        
        if args.question:
            # Query about a specific PDF in the project
            logger.info(f"Question about PDF {args.pdf}: {args.question}")
            answer, citations = pipeline.answer_question_for_pdf_in_project(args.question, args.pdf, project_id)
            
            print("\nAnswer:")
            print(answer)
            
            if citations:
                print("\nCitations:")
                for citation in citations:
                    print(f"- Page {citation}")
            
        else:
            logger.info(f"PDF {args.pdf} processed and indexed in project {project_id}. Run again with a question to query.")
    
    elif args.question and args.project:
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
