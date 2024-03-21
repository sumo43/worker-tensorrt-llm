#!/bin/bash


# Set the paths to the Mistral model, TensorRT-LLM repo, and output directories
MISTRAL_MODEL_DIR=$1
MODEL_PATH="mistralai/Mistral-7B-v0.1"
TENSORRT_LLM_DIR="/TensorRT-LLM"
OUTPUT_DIR="/volume/output"
ENGINE_DIR="/volume/engine"

ulimit -n 100000 

echo "installing tensorrt llm"
pip3 install tensorrt_llm==0.8.0 -U --pre --extra-index-url https://pypi.nvidia.com

echo "installing mpmath fix"
pip3 uninstall --yes mpmath && pip3 install mpmath==1.3.0

#git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 /volume/tokenizer

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $ENGINE_DIR

echo "cloning model repo"
apt-get install git-lfs

git lfs install

#huggingface-cli download mistralai/mistral-7b-v0.1 --local-dir /volume/tokenizer --local-dir-use-symlinks False
huggingface-cli download bigscience/bloom-560m --local-dir /volume/tokenizer --local-dir-use-symlinks False

#alias python=python3
#mkdir -p /volume/tokenizer && git clone https://huggingface.co/bigscience/bloom-560m /volume/tokenizer

# Convert the checkpoint
python3 $TENSORRT_LLM_DIR/examples/bloom/convert_checkpoint.py --model_dir "/volume/tokenizer" \
                                                              --output_dir $OUTPUT_DIR \
                                                              --dtype float16

# Build the TensorRT engine
trtllm-build --checkpoint_dir $OUTPUT_DIR \
             --output_dir $ENGINE_DIR 

