#!/bin/bash
set -e
source activate
cd .

echo "Merging SciERC 3-epoch LoRA adapter..."
echo "  Base model: ./models/Qwen/Qwen3-8B"
echo "  Adapter: checkpoints/qwen3-8b-scierc-3epoch-lora/"
echo "  Output: checkpoints/qwen3-8b-scierc-3epoch-merged/"

llamafactory-cli export export_config_scierc_3epoch.yaml

echo "Merge complete."
ls -la checkpoints/qwen3-8b-scierc-3epoch-merged/config.json
