# Start Server Script for PBM LangGraph Framework
# This script starts the FastAPI server with proper environment setup

# Exit on error
$ErrorActionPreference = "Stop"

# Configuration
$PORT = if ($env:PORT) { $env:PORT } else { 8000 }
$ServerHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$RELOAD = if ($env:RELOAD) { $env:RELOAD } else { "false" }

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Starting PBM LangGraph Framework" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check Python version (requires 3.8+)
$versionString = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$version = [version]$versionString
$requiredVersion = [version]"3.8"

if ($version -lt $requiredVersion) {
    Write-Host "Error: Python 3.8+ required, found Python $versionString" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists and activate it
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "No virtual environment found, using system Python" -ForegroundColor Yellow
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow

try {
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "Error: FastAPI not installed" -ForegroundColor Red
    Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

try {
    python -c "import uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "Error: Uvicorn not installed" -ForegroundColor Red
    Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "Dependencies check passed" -ForegroundColor Green

# Check if main.py exists
if (-not (Test-Path ".\main.py")) {
    Write-Host "Error: main.py not found in current directory" -ForegroundColor Red
    exit 1
}

# Display configuration
Write-Host ""
Write-Host "Server Configuration:" -ForegroundColor Blue
Write-Host "   Host: $ServerHost" -ForegroundColor Green
Write-Host "   Port: $PORT" -ForegroundColor Green
Write-Host "   Reload: $RELOAD" -ForegroundColor Green
Write-Host "   Working Directory: $PWD" -ForegroundColor Green
Write-Host ""

# Kill any existing process on the port
try {
    $connection = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if ($connection) {
        $processId = $connection.OwningProcess
        Write-Host "Port $PORT is already in use" -ForegroundColor Yellow
        Write-Host "Killing existing process (PID: $processId) on port $PORT..." -ForegroundColor Yellow
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        Write-Host "Process killed" -ForegroundColor Green
    }
} catch {
    # Ignore errors if port is not in use
}

# Start the server
Write-Host "Starting server..." -ForegroundColor Green
Write-Host "Server will be available at: http://${ServerHost}:${PORT}" -ForegroundColor Blue
Write-Host "Health check: http://${ServerHost}:${PORT}/health" -ForegroundColor Blue
Write-Host "API docs: http://${ServerHost}:${PORT}/docs" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the server
if ($RELOAD -eq "true") {
    # Use uvicorn directly with reload for development
    python -m uvicorn main:app --host $ServerHost --port $PORT --reload
} else {
    # Use uvicorn directly (more reliable than main.py on Windows)
    python -m uvicorn main:app --host $ServerHost --port $PORT
}

