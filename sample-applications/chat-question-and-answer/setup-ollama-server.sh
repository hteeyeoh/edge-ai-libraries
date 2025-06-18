#!/bin/sh

# Unset proxy environment variables
unset HTTP_PROXY && unset http_proxy && unset https_proxy

# Start Ollama server in the background temporarily
echo "Starting Ollama server in the background..."
ollama serve &

echo "Waiting for Ollama server to start..."
sleep 5s

# Capture PID to kill it later
OLLAMA_PID=$!

# Pull the model
echo "Pulling model ${LLM_MODEL:-tinyllama}..."
ollama pull "${LLM_MODEL:-tinyllama}"

# Kill background Ollama server (we’ll restart in foreground)
echo "Stopping background Ollama server..."
/bin/kill $OLLAMA_PID

# Run Ollama server in foreground (main container process)
echo "Starting Ollama server in foreground..."
exec ollama serve
