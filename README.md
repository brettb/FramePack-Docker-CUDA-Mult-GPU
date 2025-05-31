After geting the original FramePack Docker CUDA repository functioning, I modified it to work with the latest version of PyTorch and CUDA. I then thought it would be easy to add multi-GPU support, I was wrong. I ended up spending a bunch of time modifying the code to make it work with multiple GPUs and CUDA container, I hope to save you that time.

You can now launch FramePack with multiple GPUs by running the `run_gpu0.sh` and `run_gpu1.sh` scripts. Each GPU will utilize the same models and output directories, they will however be running on different GPUs. You can add additional GPUs by modifying the `run_gpu*.sh` scripts to your specific configuration. You can also ad the --share flag for a public URL to these scripts if desired.

You should also verify that the TORCH_CUDA_ARCH_LIST matches the architecture of your GPU. You can find this by running `nvidia-smi` on the host. The value is located in the Dockerfile

Thanks to akitaonrails for the original FramePack Docker CUDA repository. You can find it here:
https://github.com/akitaonrails/FramePack-Docker-CUDA

# FramePack Docker CUDA

Docker container for running FramePack with CUDA support, optimized for multi-GPU environments. This setup dynamically patches FramePack source code for compatibility with newer PyTorch & CUDA versions and handles Hugging Face model loading specifics.

## Prerequisites

- Docker with NVIDIA Container Toolkit installed.
- Recent NVIDIA drivers (e.g., v535.x.x or later, compatible with CUDA 12.x).
- One or more CUDA-compatible GPUs (NVIDIA).
- At least 50GB free disk space for models and Docker images.
- `git` for cloning this repository.

## Setup & Running

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/brettb/FramePack-Docker-CUDA-Multi-GPU.git
    cd FramePack-Docker-CUDA-Multi-GPU
    ```
    The necessary host-side directories (`outputs`, `hf_download`, `models`) will be created automatically by the run scripts if they don't exist and are mounted into the container.

2.  **Build the Docker Image**:
    This only needs to be done once, or when you modify `Dockerfile` or `entrypoint.sh`.
    ```bash
    docker build -t framepack-torch26-cu124:optimized .
    ```

3.  **Run FramePack**:

    *   **Single GPU (using GPU 0)**:
        ```bash
        ./run_gpu0.sh
        ```
        Access the web interface at `http://localhost:7860`.

    *   **Multi-GPU Setup**:
        For systems with multiple GPUs, you can run separate instances on different GPUs. The scripts `run_gpu0.sh` and `run_gpu1.sh` are pre-configured for GPU 0 and GPU 1 respectively.

        -   **Terminal 1 (for GPU 0)**:
            ```bash
            ./run_gpu0.sh
            ```
            Access at `http://localhost:7860`.

        -   **Terminal 2 (for GPU 1)**:
            ```bash
            ./run_gpu1.sh
            ```
            Access at `http://localhost:7861`.

        You can adapt these scripts for more GPUs by changing the `CUDA_VISIBLE_DEVICES` (on the host side for Docker) and `PORT` variables.

## Key Scripts & Configuration

-   **`Dockerfile`**: Defines the Docker image, installs system dependencies, Python environment (Python 3.12 in a venv), PyTorch 2.2.2+cu121, and FramePack. It also copies the entrypoint script and applies necessary patches to FramePack source code during the build.
-   **`entrypoint.sh`**: Script executed when the container starts. It performs:
    -   Initial diagnostic checks (`nvidia-smi`, Python environment check).
    -   Activation of the Python virtual environment.
    -   Creation and permission setup for mounted directories (`/app/outputs`, `/app/hf_download`, `/app/models`).
    -   Sets `PYTHONPATH` to include `/app` for FramePack module discovery.
    -   Launches the FramePack `demo_gradio.py` application or a bash shell.
-   **`run_gpu0.sh` / `run_gpu1.sh`**: Host-side shell scripts to easily run the Docker container on a specific GPU (0 or 1) with appropriate port mapping and environment variables. They handle mounting local directories for outputs, Hugging Face cache, and models.
-   **`check_python.py`**: A simple Python script run by the entrypoint to display Python, PyTorch, and CUDA availability details from within the container for diagnostic purposes.

## Technical Details

-   **Base Image**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
-   **Python**: 3.12 (via virtual environment in `/app/venv`)
-   **PyTorch**: 2.2.2+cu121 (compatible with CUDA 12.1, installed via pip)
-   **FramePack Patches (applied in `Dockerfile`)**:
    1.  `diffusers_helper/models/hunyuan_video_packed.py`: Replaces `torch.backends.cuda.cudnn_sdp_enabled()` with `torch.backends.cuda.flash_sdp_enabled()` to fix `AttributeError` with newer PyTorch.
    2.  `demo_gradio.py`: Modifies `AutoencoderKLHunyuanVideo.from_pretrained(...)` for the VAE to explicitly request `weight_name="diffusion_pytorch_model.safetensors"`, fixing model download issues.
-   **Container User**: Runs as `root` inside the container to manage permissions on mounted volumes and execute entrypoint tasks. Host user/group IDs are not directly used for file ownership inside due to this, but `chmod 777` is used on shared directories by the entrypoint.

## Environment Variables in Run Scripts

The `run_gpu*.sh` scripts set several environment variables for the Docker container:

-   `CUDA_VISIBLE_DEVICES` (Host): Determines which physical GPU Docker maps into the container (e.g., `1` for host GPU 1).
-   `CUDA_VISIBLE_DEVICES` (Container, via `-e`): Set to `0` inside the container. This is because Docker re-indexes the exposed GPU. So, if host GPU 1 is passed, it becomes GPU `0` from the container's perspective.
-   `PORT`: Sets the host and container port for the Gradio UI (e.g., `7860` for GPU 0, `7861` for GPU 1).
-   `PYTORCH_CUDA_ALLOC_CONF`: Configures PyTorch's CUDA memory allocator (e.g., `max_split_size_mb:64`).
-   `PYTORCH_NO_CUDA_MEMORY_CACHING=1`: Can help with memory management in some scenarios.
-   Other variables like `TORCH_CUDA_ARCH_LIST`, `TF_ENABLE_ONEDNN_OPTS` are set for potential performance tuning or compatibility, largely inherited from initial FramePack recommendations.

## First Run Notes

The first time you run a container instance, FramePack will download necessary models (HunyuanVideo, Flux, VAE, etc.). This can be over 30GB and take a considerable amount of time depending on your internet connection. These models are cached in the `./hf_download/hub` directory on your host (mounted to `/app/hf_download/hub` in the container), so subsequent runs will be much faster.

## Troubleshooting

1.  **`RuntimeError: No CUDA GPUs are available`**
    *   Ensure NVIDIA drivers are correctly installed on your host: `nvidia-smi`.
    *   Verify Docker can access GPUs: `docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 nvidia-smi`. This should list your GPUs.
    *   Check the `run_gpu*.sh` script ensures the correct host `CUDA_VISIBLE_DEVICES` is used for the `--gpus` flag.
    *   Inside the container, logs might show `cuda:0` even if you're targeting host GPU 1. This is expected, as the specific host GPU is mapped to index `0` within the container's isolated environment.

2.  **Permission Issues with Mounted Volumes**
    *   The `new_entrypoint.sh` script attempts to `chmod -R 777` the `/app/outputs`, `/app/hf_download`, and `/app/models` directories inside the container. The container runs as `root` to facilitate this.
    *   If you still face issues, ensure the host directories (`./outputs`, `./hf_download`, `./models`) are not overly restricted.

3.  **Out of Memory (OOM) Errors**
    *   FramePack can be memory-intensive. 24GB+ of GPU VRAM is recommended per instance.
    *   Try reducing batch sizes or image/video dimensions within the FramePack UI.
    *   The `PYTORCH_CUDA_ALLOC_CONF` environment variable is set to help manage memory fragmentation.

4.  **Model Download Issues (404 errors, etc.)**
    *   The patch in `Dockerfile` for `demo_gradio.py` (specifying `weight_name="diffusion_pytorch_model.safetensors"` for the VAE) should resolve common VAE download errors. If other models fail, check your internet connection and available disk space in `./hf_download`.

### Viewing Container Logs

If a container is run with `--name framepack-gpu0` (as in the scripts):
```bash
docker logs -f framepack-gpu0
# or for GPU 1 instance
docker logs -f framepack-gpu1
```
If run without a specific name and it's the last container started:
```bash
docker logs -f $(docker ps -ql)
```

### Monitoring GPU Usage

-   **On the host**: `nvidia-smi` or `nvtop`
    **Inside a running container** (e.g., if you `exec` into it):
    ```bash
    # Find container ID or name: docker ps
    docker exec -it <container_name_or_id> nvidia-smi
    ```

## License

This project's scripts and Docker configuration are provided under the MIT License. The FramePack software itself has its own licensing terms.
