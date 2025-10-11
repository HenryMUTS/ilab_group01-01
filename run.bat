@echo off
REM Start the API server in a new terminal window (keeps the window open)
start "API Server" cmd /k "cd /d API && poetry run uvicorn model_api:app --reload --host 127.0.0.1 --port 8000"

REM Start the Streamlit UI in a new terminal window (keeps the window open)
start "Streamlit UI" cmd /k "cd /d UI && poetry run streamlit run app.py"

REM Optional: keep the launcher window open so you can see messages
echo Launched API and Streamlit. Close this window to end.
pause