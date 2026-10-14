# PowerShell runner for DeepResearch AI
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  DeepResearch AI: Web Research & Intelligent Answering  " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Launching Streamlit application..." -ForegroundColor Green

$venvStreamlit = ".venv\Scripts\streamlit.exe"
if (Test-Path $venvStreamlit) {
    & $venvStreamlit run app.py
} else {
    streamlit run app.py
}
