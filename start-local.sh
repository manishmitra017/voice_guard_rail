#!/bin/bash
# Start both FastAPI backend and React frontend for local development

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Voice Emotion Detector...${NC}"

# Kill any existing processes on ports 8000 and 3000
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Start FastAPI backend
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000${NC}"
cd "$PROJECT_DIR"
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to load models..."
sleep 3

# Check if backend is running
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "  Loading models (this may take a minute on first run)..."
    sleep 5
done
echo -e "${GREEN}Backend ready!${NC}"

# Start React frontend
echo -e "${GREEN}Starting React frontend on http://localhost:3000${NC}"
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend
sleep 2

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Voice Emotion Detector is running!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  Frontend:  ${BLUE}http://localhost:3000${NC}"
echo -e "  API:       ${BLUE}http://localhost:8000${NC}"
echo -e "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    echo "Done!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
