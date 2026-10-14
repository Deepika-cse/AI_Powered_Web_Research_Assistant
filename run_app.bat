@echo off
echo =========================================================
echo   DeepResearch AI: Web Research & Intelligent Answering
echo =========================================================
echo Starting Streamlit application...

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
