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
# Balanced performance and memory settings
# Configure allocator: moderate split size, earlier GC, async backend
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True,backend:cudaMallocAsync"
# Enable PyTorch's memory caching for performance
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
# Keep model in full precision and on GPU
export MODEL_KEEP_IN_FP32=1
export TRANSFORMERS_OFFLOAD_CPU=0
export PIPELINE_OFFLOAD_CPU=0
# Enable cuDNN benchmarking for performance (finds optimal algorithms)
export CUDNN_BENCHMARK=1
# Ensure TOKENIZERS_PARALLELISM is false if set elsewhere, otherwise can be omitted if default is fine
# export TOKENIZERS_PARALLELISM=false # This was already present later, ensure consistency or remove redundancy
# OMP_NUM_THREADS and MKL_NUM_THREADS are fine at 1 for single GPU tasks if not CPU-bound for pre/post processing
# export OMP_NUM_THREADS=1 # Already present
# export MKL_NUM_THREADS=1 # Already present
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export PORT=7860 # Port for this instance
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
docker run -it --rm --name framepack-gpu0 \
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
  --ulimit memlock=1073741824 \
  --ulimit stack=67108864 \
  --ipc=host \
  --shm-size=1g \
  --user root \
  $IMAGE_NAME \
  python3 demo_gradio_f1.py --server 0.0.0.0 --port ${PORT:-7860}
  # The Docker image's ENTRYPOINT ["/app/entrypoint.sh"] will be used with the new CMD.
