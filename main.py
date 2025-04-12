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
    parser.add_argument("--question", type=str, help="Question to ask about the PDF")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the vector index")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(config)
    
    if args.pdf:
        if args.rebuild_index or not pipeline.index_exists(args.pdf):
            logger.info(f"Processing PDF: {args.pdf}")
            pipeline.process_pdf(args.pdf)
        
        if args.question:
            logger.info(f"Question: {args.question}")
            answer, citations = pipeline.answer_question(args.question, args.pdf)
            
            print("\nAnswer:")
            print(answer)
            
            if citations:
                print("\nCitations:")
                for citation in citations:
                    print(f"- Page {citation}")
        else:
            logger.info("PDF processed and indexed. Run again with a question to query the document.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
