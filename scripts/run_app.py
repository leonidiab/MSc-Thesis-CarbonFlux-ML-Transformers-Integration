#!/usr/bin/env python
"""Launch script for the NEE Carbon Flux Model Testing Dashboard.

This script starts the Flask server and automatically opens the user's
web browser to display the interactive dashboard.

Usage:
    python scripts/run_app.py
"""

import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

# Add project root to python path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from flask import Flask
except ImportError:
    print("Error: Flask is not installed in the current virtual environment.")
    print("Please install requirements using: pip install flask pandas scikit-learn joblib xgboost matplotlib seaborn")
    sys.exit(1)

from nee_classification.web.server import app

def open_browser():
    """Wait for the server to spin up, then open the default browser."""
    time.sleep(1.5)
    print("\n" + "=" * 60)
    print("Opening model testing dashboard in your default browser...")
    print("Server URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    webbrowser.open("http://127.0.0.1:5000")

def main():
    """Main entry point to run the web application."""
    print("Starting NEE Carbon Flux Testing Server...")
    
    # Check outputs directory for models
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.is_dir() or not any(outputs_dir.rglob("info.json")):
        print("\n" + "!" * 60)
        print("WARNING: No trained models detected in the 'outputs/' directory.")
        print("Please train a model first so that it is available in the dashboard.")
        print("Example training command:")
        print("  python scripts/train.py --config configs/cross_site/holdout_wombat.yaml --model trees")
        print("!" * 60 + "\n")

    # Start the browser thread
    # Only open browser if Flask is not running in reloader child process to avoid double opening
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask server
    app.run(host="127.0.0.1", port=5000, debug=True)

if __name__ == "__main__":
    main()
