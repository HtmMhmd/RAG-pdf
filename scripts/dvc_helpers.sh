#!/bin/bash

function dvc_track_project() {
    local project_id=$1
    
    if [ -z "$project_id" ]; then
        echo "Error: No project ID provided"
        return 1
    fi
    
    # Path to the project's vector database
    local vector_db_path="projects/${project_id}/vector_db"
    
    # Check if the directory exists
    if [ ! -d "$vector_db_path" ]; then
        echo "Error: Project vector database not found at $vector_db_path"
        return 1
    fi
    
    # Track with DVC
    echo "Tracking project $project_id vector database with DVC..."
    
    if ! tail -n 1 .gitignore | grep -q "^$vector_db_path/\*\*$"; then
        echo "$vector_db_path/**" >> .gitignore
    fi
    dvc add "$vector_db_path"
    
    # If auto-push is enabled and a remote is configured
    if [ "$DVC_AUTO_PUSH" = "true" ] && [ -n "$DVC_REMOTE" ]; then
        echo "Pushing project $project_id vector database to DVC remote..."
        dvc push "$vector_db_path.dvc"
    fi
    
    echo "Successfully tracked project $project_id vector database"
    return 0
}

function dvc_pull_project() {
    local project_id=$1
    
    if [ -z "$project_id" ]; then
        echo "Error: No project ID provided"
        return 1
    fi
    
    # Path to the project's vector database DVC file
    local dvc_file="projects/${project_id}/vector_db.dvc"
    
    # Check if the DVC file exists
    if [ ! -f "$dvc_file" ]; then
        echo "Warning: DVC file not found at $dvc_file"
        return 0
    fi
    
    # Pull from DVC remote
    echo "Pulling project $project_id vector database from DVC remote..."
    dvc pull "$dvc_file"
    
    echo "Successfully pulled project $project_id vector database"
    return 0
}

function dvc_setup_remote() {
    local remote_url=$1
    
    if [ -z "$remote_url" ]; then
        echo "Error: No remote URL provided"
        return 1
    fi
    
    # Check if remote is already configured
    if dvc remote list | grep -q "^default"; then
        echo "DVC remote already configured"
        return 0
    fi
    
    # Set up DVC remote
    echo "Setting up DVC remote at $remote_url..."
    dvc remote add -d myremote "$remote_url"
    
    echo "Successfully configured DVC remote"
    return 0
}

# Export functions for use in other scripts
export -f dvc_track_project
export -f dvc_pull_project
export -f dvc_setup_remote
