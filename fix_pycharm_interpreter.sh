#!/bin/bash

# Quick Fix: Install packages in PyCharm's current interpreter
# This is a TEMPORARY fix until you change PyCharm's interpreter

echo "================================"
echo "Installing packages in PyCharm's current interpreter"
echo "================================"
echo ""

PYCHARM_PYTHON="/Users/c882025/PycharmProjects/PBMAssist/venv/bin/python"
PROJECT_DIR="/Users/c882025/PycharmProjects/pss-myclaims-ai-agent"

echo "Target Python: $PYCHARM_PYTHON"
echo "Project: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

echo "Step 1: Upgrading pip..."
$PYCHARM_PYTHON -m pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

echo ""
echo "Step 2: Installing all requirements..."
$PYCHARM_PYTHON -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

echo ""
echo "Step 3: Verifying critical package..."
$PYCHARM_PYTHON -c "
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    print('✅ langgraph.checkpoint.sqlite - INSTALLED')
except ImportError as e:
    print('❌ Still missing:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ SUCCESS! You can now run the debugger in PyCharm"
    echo "================================"
    echo ""
    echo "Try debugging again!"
else
    echo ""
    echo "================================"
    echo "❌ Installation failed. See errors above."
    echo "================================"
    exit 1
fi

