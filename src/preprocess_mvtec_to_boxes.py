# This script processes the MVTec Capsule dataset to create a CSV file with bounding box annotations.
# It extracts bounding boxes from defect masks and associates them with images.

# Imports
from pathlib import Path
import pandas as pd
import cv2

# Dataset paths
root = Path('../data/capsule')
train_dir = root / 'train'
test_dir = root / 'test'
gt_dir = root / 'ground_truth'

# Function to extract defect bounding boxes from mask images
defect_classes = [p.name for p in gt_dir.iterdir()]  # e.g., ['crack', 'dent', ...]
def mask_to_boxes(mask_path, threshold=127):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return []
    _, thresh = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]  # (x, y, w, h)
    return [(x, y, x + w, y + h) for x, y, w, h in boxes]

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
            'x_min': None, 'y_min': None, 'x_max': None, 'y_max': None
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
output_path = (Path(__file__).parent.parent / 'data' / 'capsule' / 'annotations.csv')
df.to_csv(output_path, index=False)