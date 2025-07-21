import requests
import zipfile
from pathlib import Path

def download_dataset():
    """Download and extract the thyroid echography dataset from GitHub releases."""
    
    # Dataset's URL
    dataset_url = "https://github.com/gabriel-lefevre/thyroid-nodule-segmentation/releases/download/dataset-v1.0/thyroid-nodule-dataset-v1.0.zip"
    
    # Create data dir
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    dataset_path = data_dir / "dataset.zip"
    
    print("Downloading thyroid dataset...")
    response = requests.get(dataset_url)
    
    with open(dataset_path, 'wb') as f:
        f.write(response.content)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    dataset_path.unlink()
    
    print("Dataset ready!")

if __name__ == "__main__":
    download_dataset()
