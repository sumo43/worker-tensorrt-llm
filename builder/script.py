from huggingface_hub import hf_hub_download

def download_model(repo_id, local_dir):
    try:
        # Download the model files
        hf_hub_download(repo_id=repo_id, local_dir=local_dir, repo_type='model', resume_download=True)
        print(f"Model downloaded successfully to {local_dir}")
    except Exception as e:
        print(f"Error downloading the model: {str(e)}")

# Specify the repository ID and local directory
repo_id = "mistralai/Mistral-7B-v0.1"
local_dir = "/volume/tokenizer"

# Download the model
download_model(repo_id, local_dir)
