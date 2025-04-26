docker run -it --rm --gpus all -p 7860:7860 \
  -v ./outputs:/app/outputs \
  -v ./hf_download:/app/hf_download \
  framepack-torch26-cu124:latest
