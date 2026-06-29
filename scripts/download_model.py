import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "minilm_l6_v2_int8.onnx")

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path}. Skipping download.")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    logger.info("Download complete.")

if __name__ == "__main__":
    download_file(MODEL_URL, MODEL_PATH)
