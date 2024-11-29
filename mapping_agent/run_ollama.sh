#!/bin/bash


# Start the Ollama server
echo "Starting Ollama server..."
ollama serve &

# Wait for the server to initialize
echo "Waiting for Ollama server to initialize..."
sleep 10  # Adjust based on expected startup time

# Verify that the server is running
echo "Testing connection to Ollama server..."
curl -s http://localhost:11434 || {
  echo "Failed to connect to Ollama server."
  exit 1
}

echo "Ollama server is running successfully. Ready for requests!"

# Pull the Mistral model during build
ollama run mistral

# Keep the container alive after Python script execution
# Keep the container alive
tail -f /dev/null
