.PHONY: help test run install clean

help:
	@echo "Available targets:"
	@echo "  make test      - Run all endpoint tests"
	@echo "  make run       - Start the server"
	@echo "  make build     - Run tests then start server (local build)"
	@echo "  make install   - Install dependencies"
	@echo "  make clean     - Clean cache files"

test:
	@echo "🧪 Running all endpoint tests..."
	@python test_all_endpoints.py

run:
	@echo "🚀 Starting server..."
	@python main.py

build: test
	@echo "✅ Build complete: All tests passed"
	@echo "🚀 To start the server, run: make run"
	@echo "   Or: python main.py"

install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt

clean:
	@echo "🧹 Cleaning cache files..."
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete"

