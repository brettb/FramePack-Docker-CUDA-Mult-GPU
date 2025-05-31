#!/bin/bash

# Exit on error and print each command
set -ex

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Debug: Show the current directory and files
ls -la "$SCRIPT_DIR/"

# Set user and group IDs to match the host user
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Define Docker image name
IMAGE_NAME="framepack-torch26-cu124:optimized"

# Set environment variables for better performance and memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
export PORT=7861 # Port for this instance
export TORCH_CUDA_ARCH_LIST="8.6 9.0" # Match Dockerfile or remove if solely in Dockerfile
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" # Match Dockerfile or remove
export TF_ENABLE_ONEDNN_OPTS=0 # Match Dockerfile or remove
export TF_CPP_MIN_LOG_LEVEL=2 # Match Dockerfile or remove
export OMP_NUM_THREADS=1 # Match Dockerfile or remove
export TOKENIZERS_PARALLELISM=false # Match Dockerfile or remove

# Check if the Docker image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "Docker image $IMAGE_NAME not found. Building..."
  # Assuming Dockerfile is in the SCRIPT_DIR
  docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
  if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
  fi
  echo "Docker image $IMAGE_NAME built successfully."
else
  echo "Docker image $IMAGE_NAME found."
fi

# Run the Docker container with GPU support
docker run -it --rm --name framepack-gpu1 \
  --gpus device=$CUDA_VISIBLE_DEVICES \
  -e USER_ID=$USER_ID \
  -e GROUP_ID=$GROUP_ID \
  -e PORT=$PORT \
  -e PYTORCH_CUDA_ALLOC_CONF \
  -e PYTORCH_NO_CUDA_MEMORY_CACHING \
  -e TORCH_CUDA_ARCH_LIST \
  -e TORCH_NVCC_FLAGS \
  -e TF_ENABLE_ONEDNN_OPTS \
  -e TF_CPP_MIN_LOG_LEVEL \
  -e OMP_NUM_THREADS \
  -e TOKENIZERS_PARALLELISM \
  -v "${SCRIPT_DIR}/outputs:/app/outputs" \
  -v "${SCRIPT_DIR}/hf_download:/app/hf_download" \
  -v "${SCRIPT_DIR}/models:/app/models" \
  -p ${PORT}:${PORT} \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  --shm-size=16g \
  --user root \
  $IMAGE_NAME
  # The Docker image's ENTRYPOINT ["/app/entrypoint.sh"] and CMD ["app"] will be used.
