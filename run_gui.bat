@echo off
REM Guardrails Pipeline GUI Launcher
REM Run from repo root to start the Streamlit web interface

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Ensure requirements are installed
echo Installing dependencies...
pip install -q -r requirements.txt

REM Launch Streamlit app
echo.
echo Starting Guardrails Pipeline GUI...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop the server
echo.
streamlit run src/gui/app.py

pause
