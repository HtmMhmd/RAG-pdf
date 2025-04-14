#!/bin/bash
# Script for bulk processing PDFs in a project

# Get project name and PDF directory from arguments with defaults
PROJECT_NAME=${1:-"your_project"}
PDF_DIR=${2:-"projects/$PROJECT_NAME"}

echo "=== PDF Processing for Project: $PROJECT_NAME ==="
echo "Source directory: $PDF_DIR"

# Count PDFs to process
pdf_count=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
if [ $pdf_count -eq 0 ]; then
  echo "No PDF files found in $PDF_DIR"
  exit 1
fi

echo "Found $pdf_count PDF files to process"
echo "Starting batch processing..."

# Process each PDF file with proper Docker flags
processed=0
find "$PDF_DIR" -name "*.pdf" | while read -r pdf_file; do
  processed=$((processed + 1))
  echo "[$processed/$pdf_count] Processing: $pdf_file"
  
  # Use -T flag to prevent TTY allocation error
  docker-compose run --rm -T rag-pdf --pdf "$pdf_file" --project "$PROJECT_NAME" --rebuild-index
  
  if [ $? -eq 0 ]; then
    echo "✓ Completed: $pdf_file"
  else
    echo "✗ Failed to process: $pdf_file"
  fi
done

echo "=== Processing complete: $processed/$pdf_count files processed ==="