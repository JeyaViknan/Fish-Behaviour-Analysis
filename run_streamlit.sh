#!/bin/bash
# Run Streamlit app for fish behavior detection
cd "$(dirname "$0")"

PORT="${PORT:-8501}"

echo "Starting Streamlit app..."
echo "The app will open in your browser automatically."
echo "Press Ctrl+C to stop."
echo "Using port: ${PORT}"

streamlit run streamlit_app.py --server.port "${PORT}" --server.address localhost
