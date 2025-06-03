# Use CUDA 12.4.2 with cuDNN 9 for better compatibility
# FROM nvidia/cuda:12.4.2-cudnn9-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set user/group IDs to match host user (default 1000 for first user)
ARG UID=1000
ARG GID=1000

# Set environment variables for Python and system optimizations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    USER=appuser \
    # PyTorch specific optimizations
    TORCH_CUDA_ARCH_LIST="8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    # Memory management
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True" \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    # Performance optimizations
    CUDA_LAUNCH_BLOCKING=0 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Hugging Face cache settings
    HF_HOME="/app/hf_download" \
    HUGGINGFACE_HUB_CACHE="/app/hf_download/hub" \
    TRANSFORMERS_CACHE="/app/hf_download" \
    # Disable unnecessary logging
    GIT_PYTHON_REFRESH=quiet

# Create system user and group if they don't exist
RUN if ! getent group $GID >/dev/null; then groupadd -g $GID appuser; fi && \
    if ! id -u $UID >/dev/null 2>&1; then useradd -u $UID -g $GID -m -s /bin/bash appuser; fi

# Install system dependencies with optimizations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    # Graphics libraries
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # AVIF support for PIL/Pillow
    libavif-dev \
    # Build tools
    ninja-build \
    build-essential \
    cmake \
    sudo \
    # Performance monitoring
    htop \
    nvtop \
    # System utils
    wget \
    # FlashAttention2 build dependencies
    pkg-config \
    clang \
    libomp-dev \
    g++ \
    gcc \
    make \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "appuser ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers \
    # Create a clean cache directory that can be written to by the user
    && mkdir -p /home/appuser/.cache \
    && chown -R $UID:$GID /home/appuser/.cache && \
    # Install ninja-build from package manager
    apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and configure directories before switching user
RUN mkdir -p /app && \
    chown -R $UID:$GID /app

# Copy entrypoint and check_python script before switching user
COPY check_python.py /app/check_python.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_python.py && \
    chown $UID:$GID /app/entrypoint.sh /app/check_python.py

# Switch to non-root user
USER $UID:$GID

# Clone repository
RUN TEMP_CLONE_DIR=$(mktemp -d) && \
    git clone https://github.com/brettb/FramePack.git "$TEMP_CLONE_DIR" && \
    cp -a "$TEMP_CLONE_DIR"/. /app/ && \
    rm -rf "$TEMP_CLONE_DIR"

# Apply runtime patches during build
RUN sed -i 's/torch.backends.cuda.cudnn_sdp_enabled()/torch.backends.cuda.flash_sdp_enabled()/g' "/app/diffusers_helper/models/hunyuan_video_packed.py" && \
    sed -i 's/subfolder='\''vae'\'', torch_dtype=torch.float16\\)/subfolder='\''vae'\'', weight_name="diffusion_pytorch_model.safetensors", torch_dtype=torch.float16)/g' "/app/demo_gradio.py"
WORKDIR /app

# Create virtual environment as user
RUN python3 -m venv $VIRTUAL_ENV

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.4 runtime)
# RUN pip install --no-cache-dir -U pip setuptools wheel && \
#    pip install --no-cache-dir \
#    --index-url https://download.pytorch.org/whl/cu128 \
#    torch==2.7.0+cu128 \
#    torchvision \
#    torchaudio \
#    --no-cache-dir

# Install requirements with optimizations
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --prefer-binary \
    --no-build-isolation

# Install additional dependencies with optimizations
RUN pip install --no-cache-dir \
    triton==3.0.0 \
    ninja \
    nvidia-ml-py3 \
    py-cpuinfo \
    psutil \
    py3nvml \
    wheel \
    --prefer-binary

# Install Framepack source in the main virtual environment
RUN set -x && \
    # Set CUDA environment variables needed for compilation
    export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 15.0+PTX" && \
    export SAGEATTENTION_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;15.0" && \
    export FORCE_CUDA=1 && \
    export CMAKE_CUDA_ARCHITECTURES="80;86;89;90;150" && \
    export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80;86;89;90;150" && \
    export MAX_JOBS=$(nproc) && \
    # Clone the repository
    git clone --depth 1 https://github.com/brettb/FramePack.git /tmp/FramePack && \
    cd /tmp/FramePack && \
    # Show directory structure and look for setup.py files (especially for SageAttention)
    echo "=== Directory structure in /tmp/FramePack ===" && \
    ls -la && \
    echo "\n=== Searching for setup.py files ===" && \
    find . -name "setup.py" -print && \
    # Show requirements.txt content
    echo "\n=== requirements.txt content ===" && \
    cat requirements.txt && \
    # Install required dependencies from requirements.txt
    # The CUDA environment variables set above will apply to this step.
    echo "\n=== Installing requirements from requirements.txt ===" && \
    pip install -r requirements.txt --verbose && \
    # At this point, dependencies are installed. FramePack scripts are available in /tmp/FramePack.
    # SageAttention might have been installed if it was in requirements.txt, or if it's a sub-module,
    # we'll need to install it explicitly based on the 'find setup.py' output.
    # For now, we are NOT running 'pip install .' at the root of FramePack.
    echo "\n=== FramePack dependencies installed. SageAttention build (if separate) needs to be confirmed. ===" && \
    # Verify Python environment (optional)
    echo "\n=== Verifying Python environment ===" && \
    python -c "import sys; import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('\nPython sys.path:'); print('\n'.join(sys.path))" && \
    # Clean up /tmp/FramePack. 
    # WARNING: If FramePack scripts are run directly from this path later, do not remove it yet.
    # Assuming for now that necessary components are installed into site-packages or copied elsewhere.
    cd /app && \
    rm -rf /tmp/FramePack && \
    echo "\n=== Docker build step for FramePack completed ==="

# Install performance monitoring tools and AVIF support for Pillow
RUN pip install --no-cache-dir setuptools>=45 && \
    pip install --no-cache-dir \
    nvitop \
    gpustat \
    py3nvml \
    pillow-avif-plugin \
    --prefer-binary

# Create application directories
RUN mkdir -p /app/outputs /app/hf_download /app/models

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    HOME=/app

# Switch to non-root user (ensure this is after chown)
USER $UID:$GID

# Create any additional directories as the non-root user if needed
# RUN mkdir -p /app/outputs /app/hf_download /app/models # These are already created by root and chowned

# Expose ports and volumes
EXPOSE 7860
VOLUME ["/app/outputs", "/app/hf_download", "/app/models"]

# Set the entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["app"]
