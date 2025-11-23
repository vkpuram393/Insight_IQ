#!/bin/bash

# Start Server Script for PBM LangGraph Framework
# This script starts the FastAPI server with proper environment setup

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
RELOAD="${RELOAD:-false}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Starting PBM LangGraph Framework${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 is not installed${NC}"
    exit 1
fi

# Check Python version (requires 3.8+)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Error: Python 3.8+ required, found Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version: $(python3 --version)${NC}"

# Check if virtual environment exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
    source "$SCRIPT_DIR/venv/bin/activate"
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment found, using system Python${NC}"
fi

# Check if requirements are installed
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}❌ Error: FastAPI not installed${NC}"
    echo -e "${YELLOW}💡 Run: pip install -r requirements.txt${NC}"
    exit 1
fi

if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}❌ Error: Uvicorn not installed${NC}"
    echo -e "${YELLOW}💡 Run: pip install -r requirements.txt${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dependencies check passed${NC}"

# Check if main.py exists
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Display configuration
echo ""
echo -e "${BLUE}📋 Server Configuration:${NC}"
echo -e "   Host: ${GREEN}$HOST${NC}"
echo -e "   Port: ${GREEN}$PORT${NC}"
echo -e "   Reload: ${GREEN}$RELOAD${NC}"
echo -e "   Working Directory: ${GREEN}$SCRIPT_DIR${NC}"
echo ""

# Kill any existing process on the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port $PORT is already in use${NC}"
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}🔪 Killing existing process (PID: $PID) on port $PORT...${NC}"
        kill -9 $PID 2>/dev/null || true
        # Wait a moment for the port to be released
        sleep 1
        echo -e "${GREEN}✅ Process killed${NC}"
    fi
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Start the server
echo -e "${GREEN}🚀 Starting server...${NC}"
echo -e "${BLUE}📍 Server will be available at: http://$HOST:$PORT${NC}"
echo -e "${BLUE}📍 Health check: http://$HOST:$PORT/health${NC}"
echo -e "${BLUE}📍 API docs: http://$HOST:$PORT/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the server
if [ "$RELOAD" = "true" ]; then
    # Use uvicorn directly with reload for development
    python3 -m uvicorn main:app --host "$HOST" --port "$PORT" --reload
else
    # Use main.py (which has its own uvicorn.run call)
    python3 main.py
fi

