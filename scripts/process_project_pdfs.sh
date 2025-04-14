#!/bin/bash
# filepath: /workspaces/RAG-pdf/scripts/process_project_pdfs.sh

# Get script location and parent (project root) directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Get project name and PDF directory from arguments with defaults
PROJECT_NAME=${1:-"your_project"}
PDF_DIR="${ROOT_DIR}/pdfs/${PROJECT_NAME}"
MAIN_SCRIPT="${ROOT_DIR}/main.py}"

echo "=== PDF Processing for Project: $PROJECT_NAME ==="
echo "Root directory: $ROOT_DIR"
echo "PDF Source directory: $PDF_DIR"

# Count PDFs to process
pdf_count=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
if [ $pdf_count -eq 0 ]; then
  echo "No PDF files found in $PDF_DIR"
  exit 1
fi

echo "Found $pdf_count PDF files to process"
echo "Starting batch processing..."

# Process each PDF file with proper Docker paths
processed=0
find "$PDF_DIR" -name "*.pdf" | while read -r pdf_file; do
  processed=$((processed + 1))
  
  # Calculate the Docker path by replacing host path with Docker container path
  # Get the relative path from ROOT_DIR
  rel_path=$(realpath --relative-to="$ROOT_DIR" "$pdf_file")
  docker_pdf_path="/app/${rel_path}"
  
  echo "[$processed/$pdf_count] Processing: $pdf_file"
  echo "  → Docker path: $docker_pdf_path"
  
  # Use -T flag to prevent TTY allocation error
  docker-compose run --rm -T rag-pdf --pdf "$docker_pdf_path" --project "$PROJECT_NAME" --rebuild-index
  
  if [ $? -eq 0 ]; then
    echo "✓ Completed: $pdf_file"
  else
    echo "✗ Failed to process: $pdf_file"
  fi
done

echo "=== Processing complete: $processed/$pdf_count files processed ==="