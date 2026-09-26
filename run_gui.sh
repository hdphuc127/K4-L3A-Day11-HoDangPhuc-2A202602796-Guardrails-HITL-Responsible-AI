#!/bin/bash
# Guardrails Pipeline GUI Launcher
# Run from repo root to start the Streamlit web interface

cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Ensure requirements are installed
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Launch Streamlit app
echo ""
echo "Starting Guardrails Pipeline GUI..."
echo "Open http://localhost:8501 in your browser"
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run src/gui/app.py
