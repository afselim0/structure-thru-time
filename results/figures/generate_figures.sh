#!/bin/bash

# Set the directory containing the Python scripts
SCRIPT_DIR="./scripts"

# Run every Python script in the directory
for script in "$SCRIPT_DIR"/*.py; do
    echo "Running $script..."
    python "$script"
done