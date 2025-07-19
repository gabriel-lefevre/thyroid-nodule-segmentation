# thyroid-nodule-segmentation
🏥 University project: Thyroid nodule segmentation on ultrasound images using U-Net (ResNet-50/VGG16) with PyTorch, complete MLOps pipeline and Flask web interface for medical AI demonstration

## Installation

### Prerequisites
- Python 3.12
- Conda

### Step 1: Clone the repository
```bash
git clone https://github.com/gabriel-lefevre/thyroid-nodule-segmentation
cd thyroid-nodule-segmentation
```

### Step 2: Create conda environment
```bash
conda create -n thyroid-nodule-segmentation python=3.12
```

### Step 3: Activate environment
```bash
conda activate thyroid-nodule-segmentation
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Download dataset
```bash
python get_dataset.py
```
### Or manually: Download from Releases > dataset-v1.0.zip
Go to [Releases](https://github.com/gabriel-lefevre/thyroid-nodule-segmentation/releases) and download `thyroid-nodule-dataset-v1.0.zip`

Extract the dataset to a `data/` folder in your project root.