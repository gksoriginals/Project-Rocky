#!/bin/bash

# Configuration
DEFAULT_MODEL="gemma4:e2b"
DB_PATH="rocky_memory.sqlite3"
ENV_FILE=".env"

echo "=== Rocky Agent Setup ==="

# 1. Environment File
echo -e "\n[1/3] Checking environment configuration..."
if [ ! -f "$ENV_FILE" ]; then
    echo "OLLAMA_MODEL=$DEFAULT_MODEL" > "$ENV_FILE"
    echo "OLLAMA_HOST=http://localhost:11434" >> "$ENV_FILE"
    echo "Created default $ENV_FILE with model $DEFAULT_MODEL"
else
    echo "$ENV_FILE already exists."
fi

# 2. Database Initialization
echo -e "\n[2/3] Initializing database: $DB_PATH"
if [ ! -f "$DB_PATH" ]; then
    # Use a python one-liner to trigger the schema creation via the existing class
    python3 -c "from rocky.memory.db import MemoryDB; db = MemoryDB('$DB_PATH'); db.close(); print('Database file created and schema applied.')"
else
    echo "Database file $DB_PATH already exists."
fi

# 3. Ollama Model
echo -e "\n[3/3] Ensuring Ollama model '$DEFAULT_MODEL' is available..."
if command -v ollama &> /dev/null; then
    echo "Pulling model (this may take a few minutes)..."
    ollama pull "$DEFAULT_MODEL"
    echo "Successfully ensured $DEFAULT_MODEL is available."
else
    echo "Warning: 'ollama' command not found. Please install Ollama from https://ollama.com"
fi

echo -e "\n=== Setup Complete ==="
echo "You can now start Rocky by running:"
echo "  poetry run python rocky.py"
