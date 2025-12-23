@echo off
echo Starting AI Service...
cd ai-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause