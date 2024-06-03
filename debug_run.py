import numpy as np
import datumaro as dm
import json

# Load COCO JSON data
coco_annotation = """
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Roboflow",
        "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
        "date_created": "2000-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Workers",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "head",
            "supercategory": "Workers"
        },
        {
            "id": 2,
            "name": "helmet",
            "supercategory": "Workers"
        },
        {
            "id": 3,
            "name": "person",
            "supercategory": "Workers"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
"""

coco_data = json.loads(coco_annotation)

# Create Datumaro categories
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
datumaro_categories = ["Workers", "head", "helmet", "person"]

items = []
for image_info in coco_data['images']:
    image_id = image_info['id']
    annotations = [
        dm.Bbox(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3], label=ann['category_id'])
        for ann in coco_data['annotations']
        if ann['image_id'] == image_id
    ]
    
    items.append(dm.DatasetItem(
        id=image_info['file_name'],
        subset='train',
        media=dm.Image(data=np.ones((image_info['height'], image_info['width'], 3))),  # Placeholder image, replace with actual image if available
        annotations=annotations
    ))

dataset = dm.Dataset(items, categories=datumaro_categories, media_type=dm.Image)

# Export Datumaro dataset to COCO format (if needed)
dataset.export('test_dataset/', 'yolov8')