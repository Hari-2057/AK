#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "Checking environment..."

# Wait up to 300 seconds for installation to finish
for i in {1..60}; do
    if command -v streamlit &> /dev/null; then
        echo "Launching App..."
        streamlit run app.py
        exit 0
    fi
    echo "First-time setup is still running (installing AI libraries)... Waiting... ($i/60)"
    sleep 5
done

echo "Error: Installation timed out or failed. Please check the logs."
exit 1
