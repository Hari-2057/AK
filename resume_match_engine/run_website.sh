#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
# Try running via python module to avoid PATH issues
python3 -m streamlit run app.py
