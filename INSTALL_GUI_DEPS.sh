#!/bin/bash

# Install PyQt5 GUI dependencies for BTC Predictor

echo "Installing PyQt5 GUI dependencies..."

# Core PyQt5
pip install PyQt5 PyQtChart PyQtWebEngine

# Plotting
pip install plotly kaleido

# Data handling (if not already installed)
pip install pandas numpy scikit-learn torch

echo "Installation complete!"
echo ""
echo "To run the GUI app:"
echo "  python gui_app.py"
