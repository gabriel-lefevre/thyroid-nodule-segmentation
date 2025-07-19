# thyroid-nodule-segmentation
🏥 University project: Thyroid nodule segmentation on ultrasound images using U-Net (ResNet-50/VGG16) with PyTorch, complete MLOps pipeline and Flask web interface for medical AI demonstration

## Installation

### Prerequisites
- Python 3.12
- Conda

### Setup
```bash
# 1. Clone repository
git clone https://github.com/gabriel-lefevre/thyroid-nodule-segmentation
cd thyroid-nodule-segmentation

# 2. Create environment
conda create -n thyroid-nodule-segmentation python=3.12
conda activate thyroid-nodule-segmentation
pip install -r requirements.txt

# 3. Download dataset
python download_dataset.py
# OR manually: Download from Releases > dataset-v1.0.zip
