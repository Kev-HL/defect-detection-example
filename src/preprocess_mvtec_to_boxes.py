"""
This script processes the MVTec Capsule dataset to create a CSV file with bounding box annotations.
It saves the bounding boxes as sets of normalized coordinates (x_min, y_min, x_max, y_max) for each defect class.
It encodes the absence of box in the 'good' class with -1 for all coordinates.
"""

# Imports
from pathlib import Path
import pandas as pd
import cv2

# Dataset paths
root = Path(__file__).parent.parent / 'data' / 'capsule'
train_dir = root / 'train'
test_dir = root / 'test'
gt_dir = root / 'ground_truth'
output_path = root / 'annotations.csv' # Output CSV file

# Definitions
defect_classes = [p.name for p in gt_dir.iterdir()]  # There is one folder per defect class
PIXEL_MAX = 255  # Max pixel value for grayscale images
PIXEL_THRESHOLD = PIXEL_MAX // 2  # Default threshold for binary mask ~ 255/2

# Function to extract defect bounding boxes from mask images
def mask_to_boxes(mask_path, threshold=PIXEL_THRESHOLD):
    """
    Extracts bounding boxes from a binary mask image.
    Returns a list of normalized (x_min, y_min, x_max, y_max) tuples.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        print(f"Warning: Mask not found or empty for {mask_path}")
        return []
    height, width = mask.shape
    _, thresh = cv2.threshold(mask, threshold, PIXEL_MAX, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]  # (x, y, w, h)
    return [
            (x / width, y / height, (x + w) / width, (y + h) / height)
            for x, y, w, h in boxes
        ]

def main():
    """
    Main function to process the dataset and create the CSV file.
    """
    # Empty list to hold records
    records = []

    # First, handle good examples (train + test /good)
    for split in ['train', 'test']:
        good_dir = root / split / 'good'
        for img_path in good_dir.glob('*.png'):
            records.append({
                'image_path': str(img_path),
                'set': split,  # 'train' or 'test'
                'class_name': 'good',
                'x_min': -1, 'y_min': -1, 'x_max': -1, 'y_max': -1
            })

    # Second, handle defect examples from test set
    for defect_class in defect_classes:
        defect_img_dir = test_dir / defect_class
        mask_dir = gt_dir / defect_class

        for img_path in defect_img_dir.glob('*.png'):
            img_name = img_path.stem  # '001'
            mask_path = mask_dir / f"{img_name}_mask.png"

            boxes = mask_to_boxes(mask_path)

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                records.append({
                    'image_path': str(img_path),
                    'set': 'test',
                    'class_name': defect_class,
                    'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max
                })

    # Convert records to DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()