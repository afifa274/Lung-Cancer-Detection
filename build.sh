#!/usr/bin/env bash
# build.sh — runs once during Render build phase
# Retrains the model so it matches the exact TF/Keras version on Render

set -e
echo "==> Retraining model for current TF/Keras version..."
python model.py
echo "==> Model retrain complete."
