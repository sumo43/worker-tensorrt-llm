#!/bin/bash

chmod +x build_llama_engine.sh
./build_llama_engine.sh

echo "Starting Triton server"
ulimit -n 1000000
tritonserver --model-repository /triton_model_repo &

echo "Starting RunPod Handler"
python3 -u /handler.py
