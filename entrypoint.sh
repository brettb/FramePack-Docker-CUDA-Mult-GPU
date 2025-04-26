#!/bin/bash
set -e

# Ensure directories have correct ownership
if [ "$(stat -c '%u' /app/outputs)" != "$UID" ]; then
    echo "Fixing permissions for /app/outputs"
    sudo chown -R $UID:$GID /app/outputs
fi

if [ "$(stat -c '%u' /app/hf_download)" != "$UID" ]; then
    echo "Fixing permissions for /app/hf_download"
    sudo chown -R $UID:$GID /app/hf_download
fi

# Check for required models
MODEL_CHECKLIST=(
  "/app/models/checkpoints/sd_xl_base_1.0.safetensors"
  "/app/models/upscale_models/realesr-general-x4v3.pth"
)

for model in "${MODEL_CHECKLIST[@]}"; do
  if [ ! -f "$model" ]; then
    echo "ERROR: Missing required model: $model"
    echo "Either mount volumes or rebuild image with model downloads"
    exit 1
  fi
done

exec "$@"
