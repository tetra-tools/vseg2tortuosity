#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the work directory inside the container; allow over-riding via environment variable
WORKDIR="${WORKDIR:-/app/workdir}"

# Log file
LOG_FILE="$WORKDIR/pipeline.log"

# Redirect all output to the log file
exec > >(tee -i "$LOG_FILE") 2>&1

# Ensure the working directory exists
mkdir -p "$WORKDIR"

# Run the main pipeline command
echo "Starting the main pipeline..."
python -m src.graph_processing.main --workdir "$WORKDIR" -v

# Clean up .npy files after generating the CSV
echo "Cleaning up .npy files..."
find "$WORKDIR" -type f -name "*.npy" -exec rm {} \;

# Ensure the 'plots' directory exists
mkdir -p "$WORKDIR/plots"

# Move all generated png plot and html files to the 'plots' directory
echo "Moving plot files to the 'plots' directory..."
find "$WORKDIR" -maxdepth 1 -type f -name "*.png" -exec mv {} "$WORKDIR/plots/" \;
find "$WORKDIR" -maxdepth 1 -type f -name "*.html" -exec mv {} "$WORKDIR/plots/" \;

echo "Pipeline and cleanup completed successfully."