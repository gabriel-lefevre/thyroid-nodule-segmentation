"""
Dataset Splitter for U-Net Training

Splits processed images and masks into train/validation/test sets
with configurable ratios. Creates organized directory structure
for machine learning model training.

Author: Gabriel Lefevre
"""
import os
import shutil
import random

from config import *


def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split cropped images and masks into train/validation/test sets.

    Creates organized dataset structure with configurable split ratios.
    Moves files from processing directories to final training structure.

    Args:
        train_ratio (float): Percentage for training set (default: 0.7)
        val_ratio (float): Percentage for validation set (default: 0.15)
        test_ratio (float): Percentage for test set (default: 0.15)
        random_seed (int): Random seed for reproducible splits (default: 42)

    Returns:
        bool: True if split completed successfully
    """
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        log(f"❌ Split ratios must sum to 1.0, got {total_ratio}")
        return False

    # Set random seed for reproducible splits
    random.seed(random_seed)

    log("Starting dataset split...")
    log(f"   Split ratios: {train_ratio:.1%} train / {val_ratio:.1%} val / {test_ratio:.1%} test")

    # Create dataset structure
    _create_dataset_structure()

    # Get matching image-mask pairs
    matching_files = _get_matching_files()

    # Split files into sets
    file_splits = _split_files(matching_files, train_ratio, val_ratio, test_ratio)

    # Move files to respective directories
    stats = _move_files_to_sets(file_splits)

    # Display final statistics
    _display_split_summary(stats)

    return True


def _create_dataset_structure():
    """
    Create organized directory structure for U-Net training.
    """
    log("Creating dataset_unet directory structure...")

    splits = ["train", "validation", "test"]
    subdirs = ["images", "masks"]

    for split in splits:
        for subdir in subdirs:
            dir_path = os.path.join(DATASET_UNET_DIR, split, subdir)
            os.makedirs(dir_path, exist_ok=True)


def _get_matching_files():
    """
    Get image files and verify they match mask files.

    Returns:
        list: Filenames of images with matching masks
    """
    image_files = [f for f in os.listdir(CROPPED_IMGS_DIR) if f.lower().endswith('.jpg')]
    mask_files = [m for m in os.listdir(MASKS_DIR) if m.lower().endswith('.jpg')]

    # Sort both lists to ensure matching order
    image_files.sort()
    mask_files.sort()

    # Verify same count
    if len(image_files) != len(mask_files):
        log(f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks")
        exit(1)

    # Verify same filenames
    if image_files != mask_files:
        log("Image and mask filenames don't match!")
        exit(1)

    return image_files


def _split_files(files, train_ratio, val_ratio, test_ratio):
    """
    Split files into train/validation/test sets with specified ratios.

    Args:
        files: List of filenames to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        dict: File splits organized by set name
    """
    log("Splitting files into train/validation/test sets...")

    # Shuffle files for random distribution
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)

    total_files = len(shuffled_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    # Split files
    train_files = shuffled_files[:train_count]
    val_files = shuffled_files[train_count:train_count + val_count]
    test_files = shuffled_files[train_count + val_count:]

    return {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }


def _move_files_to_sets(file_splits):
    """
    Move images and masks to their respective train/val/test directories.

    Args:
        file_splits: Dictionary of file splits by set name

    Returns:
        dict: Statistics of files moved per set
    """
    log("Moving files to dataset directories...")

    stats = {'train': 0, 'validation': 0, 'test': 0}

    for split_name, files in file_splits.items():
        images_dir = os.path.join(DATASET_UNET_DIR, split_name, "images")
        masks_dir = os.path.join(DATASET_UNET_DIR, split_name, "masks")

        for filename in files:
            try:
                # Move image file
                src_img = os.path.join(CROPPED_IMGS_DIR, filename)
                dst_img = os.path.join(images_dir, filename)
                shutil.move(src_img, dst_img)

                # Move corresponding mask file
                src_mask = os.path.join(MASKS_DIR, filename)
                dst_mask = os.path.join(masks_dir, filename)
                shutil.move(src_mask, dst_mask)

                stats[split_name] += 1

            except Exception as e:
                log(f"Error moving {filename}: {e}")
                continue

    return stats


def _display_split_summary(stats):
    """
    Display final split statistics and directory information.

    Args:
        stats: Dictionary of file counts per split
    """
    total_files = sum(stats.values())

    log("\n📊 DATASET SPLIT COMPLETED")
    log("=" * 50)
    log(f"Training set:     {stats['train']:4d} files ({stats['train'] / total_files:.1%})")
    log(f"Validation set:   {stats['validation']:4d} files ({stats['validation'] / total_files:.1%})")
    log(f"Test set:         {stats['test']:4d} files ({stats['test'] / total_files:.1%})")
    log(f"Total processed:  {total_files:4d} files")
    log("=" * 50)
    log(f"Dataset location: {DATASET_UNET_DIR}")
    log("\nDirectory structure:")
    log("dataset_unet/")
    log("├── train/")
    log("│   ├── images/")
    log("│   └── masks/")
    log("├── validation/")
    log("│   ├── images/")
    log("│   └── masks/")
    log("└── test/")
    log("    ├── images/")
    log("    └── masks/")


