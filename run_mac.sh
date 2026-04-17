#!/bin/bash
# Setup and run GPT-2 fine-tune on Mac (M-series, uses Metal/MPS)

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Starting fine-tune (output -> training.log)..."
python -u finetune_gpt2.py > training.log 2>&1 &
echo "Training running in background (PID $!)"
echo "Monitor with: tail -f training.log"
