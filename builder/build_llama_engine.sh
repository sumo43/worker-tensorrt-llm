

# Set the paths to the Mistral model, TensorRT-LLM repo, and output directories
MISTRAL_MODEL_DIR=$1
MODEL_PATH="mistralai/Mistral-7B-v0.1"
TENSORRT_LLM_DIR="/TensorRT-LLM"
OUTPUT_DIR="/volume/output"
ENGINE_DIR="/volume/engine"

ulimit -n 1000000

#echo "installing tensorrt llm"
#pip3 install mpi4py
#pip3 install tensorrt_llm==0.8.0 -U --pre --extra-index-url https://pypi.nvidia.com

#echo "installing mpmath fix"
#pip3 uninstall --yes mpmath && pip3 install mpmath==1.3.0

#git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 /volume/tokenizer

# Create output directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $ENGINE_DIR

echo "cloning model repo"
apt-get install git-lfs
git lfs install

echo "downloading model"
#huggingface-cli download georgesung/llama2_7b_chat_uncensored --local-dir /volume/tokenizer --local-dir-use-symlinks False  

#RUN mkdir -p /volume/tokenizer && git clone https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b /volume/tokenizer
huggingface-cli download sumo43/mistral-7b-1-H100-fp8 --local-dir /volume/engine --local-dir-use-symlinks False

#huggingface-cli download bigscience/bloom-560m --local-dir /volume/tokenizer --local-dir-use-symlinks False

#
#alias python=python3
#mkdir -p /volume/tokenizer && git clone https://huggingface.co/bigscience/bloom-560m /volume/tokenizer


# Convert the checkpoint
#python3 $TENSORRT_LLM_DIR/examples/llama/convert_checkpoint.py --model_dir "/volume/tokenizer" \
#                                                              --output_dir $OUTPUT_DIR \

ulimit -n 1000000

# Build the TensorRT engine
#trtllm-build --checkpoint_dir $OUTPUT_DIR \
#             --output_dir $ENGINE_DIR 
