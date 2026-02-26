#!/bin/bash
cd "$(dirname "$0")"

# Check if lite_venv exists, if not create it
if [ ! -d "lite_venv" ]; then
    echo "Setting up lightweight environment..."
    python3 -m venv lite_venv
    source lite_venv/bin/activate
    pip install streamlit
else
    source lite_venv/bin/activate
fi

echo "Launching Website (Lite Mode)..."
python3 -m streamlit run app.py
