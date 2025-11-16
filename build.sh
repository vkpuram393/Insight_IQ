#!/bin/bash
# Build script that runs tests before starting server

set -e  # Exit on error

echo "🔨 Starting local build..."
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Run tests
echo "🧪 Running endpoint tests..."
python test_all_endpoints.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "🚀 Starting server..."
    echo ""
    python main.py
else
    echo ""
    echo "❌ Tests failed! Not starting server."
    exit 1
fi

