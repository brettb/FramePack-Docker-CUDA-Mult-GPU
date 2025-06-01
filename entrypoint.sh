#!/bin/bash
set -e

# Create necessary directories if they don't exist and set permissions
mkdir -p /app/outputs /app/hf_download /app/models
chmod -R 777 /app/outputs /app/hf_download /app/models

echo "=== GPU Information ==="
nvidia-smi

# Activate virtual environment if it exists and is not already active
if [ -f "/app/venv/bin/activate" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source /app/venv/bin/activate
fi

echo "Current Python: $(which python3)"
echo "Current VENV: $VIRTUAL_ENV"

echo -e "\n=== Python and PyTorch Check (in venv) ==="
python3 /app/check_python.py

# Execute the command passed to the entrypoint, or default to FramePack demo
if [ "$#" -eq 0 ] || [ "$1" = "app" ]; then
    echo "Listing /app contents before cd:"
    ls -la /app
    cd /app # Ensure we are in the correct directory
    echo "Current working directory: $(pwd)"
    echo "Listing /app contents after cd:"
    ls -la /app
    echo "Listing /app/framepack contents (if it exists):"
    ls -la ./framepack || echo "/app/framepack directory not found"

    echo "Ensuring /app/hf_download/hub exists and checking permissions..."
    echo "State of /app/hf_download before mkdir:"
    ls -ld /app/hf_download || echo "/app/hf_download not found"
    ls -la /app/hf_download
    
    mkdir -p /app/hf_download/hub
    if [ $? -ne 0 ]; then
        echo "Error: mkdir -p /app/hf_download/hub failed. 'hub' might be a file."
        echo "Contents of /app/hf_download:"
        ls -la /app/hf_download
        exit 1
    fi
    # chown root:root /app/hf_download/hub # Running as root, so root will own new dirs
    # chmod 777 /app/hf_download/hub # mkdir -p creates with default umask, root can write
    
    echo "State of /app/hf_download/hub after mkdir -p:"
    ls -ld /app/hf_download/hub || echo "/app/hf_download/hub could not be verified"
    echo "Contents of /app/hf_download now:"
    ls -la /app/hf_download
    echo "Contents of /app/hf_download/hub now:"
    ls -la /app/hf_download/hub

    echo "Current PYTHONPATH: $PYTHONPATH"
    echo "Setting PYTHONPATH to include /app"
    if [ -z "$PYTHONPATH" ]; then
      export PYTHONPATH="/app"
    else
      export PYTHONPATH="/app:$PYTHONPATH"
    fi
    echo "New PYTHONPATH: $PYTHONPATH"


riginal method
#     echo "Starting FramePack application with 'python3 demo_gradio.py' ..."
#     exec python3 demo_gradio.py --server 0.0.0.0 --port ${PORT:-7860}

    echo "Starting FramePack application with 'python3 demo_gradio_f1.py' ..."
    exec python3 demo_gradio_f1.py --server 0.0.0.0 --port ${PORT:-7860}
elif [ "$1" = "bash" ]; then
    echo "Starting bash shell..."
    exec /bin/bash
else
    echo "Executing command: $@"
    exec "$@"
fi
