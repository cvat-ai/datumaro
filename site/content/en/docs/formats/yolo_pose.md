---
title: 'YOLO-Pose'
linkTitle: 'YOLO-Pose'
description: ''
---

## Format specification

The YOLO-Pose dataset format is tailored for training and validating pose estimation models using the YOLO framework. Detailed specifications for this format can be found in the official [YOLO Pose documentation](https://docs.ultralytics.com/datasets/pose/).

Supported annotation types:
- `Keypoints`

The format supports various subset names, except `classes`, `names`, and `backup`.

> Note that by default, the YOLO framework does not expect any subset names except `train` and `valid`. Datumaro supports this as an extension. If there is no subset separation in a project, the data will be saved in the `train` subset.

## Import YOLO-Pose dataset

To create a Datumaro project with a YOLO-Pose source, use the following commands:

```bash
datum create
datum import --format yolo_pose <path/to/dataset>
```

The YOLO-Pose dataset directory should have the following structure:

```bash
└─ yolo_pose_dataset/
   │
   ├── data.yaml       # file with dataset information
   ├── train.txt       # list of image paths in train subset [Optional]
   ├── valid.txt       # list of image paths in valid subset [Optional]
   │
   ├── images/
   │   ├── train/      # directory with images for train subset
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── valid/      # directory with images for valid subset
   │       ├── image11.jpg
   │       ├── image12.jpg
   │       └── ...
   │
   ├── labels/
       ├── train/      # directory with annotations for train subset
       │   ├── image1.txt
       │   ├── image2.txt
       │   └── ...
       └── valid/      # directory with annotations for valid subset
           ├── image11.txt
           ├── image12.txt
           └── ...
```

`data.yaml` should have the following content. It is not necessary to have both subsets, but necessary to have one of them. Additionally, class numbers should be zero-indexed (start with 0):

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./ # dataset root dir
train: train.txt  # train images (relative to 'path')
val: valid.txt    # val images (relative to 'path')
test: # test images (optional)

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)

# Classes
names:
  0: person
  # Add more classes if required
```

Files `train.txt` and `valid.txt` should have the following structure:

```txt
<path/to/image1.jpg>
<path/to/image2.jpg>
...
```

Files in directories `labels/train/` and `labels/valid/` should contain information about labeled keypoints for images. If there are no objects in an image, no `.txt` file is required:

```txt
# labels/train/image1.txt:
# <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>
0 0.600000 0.400000 0.400000 0.266667 0.400000 0.500000 1 0.420000 0.530000 2 ... 0.480000 0.620000 1
...
```
Each row in these label files contains the following information about the object instance:
- Object class index: An integer representing the class of the object (e.g., 0 for person, 2 for car, etc.).
- Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1.
- Object width and height: The width and height of the object, normalized to be between 0 and 1.
- Object Keypoints: The Keypoints must be in normalized coordinates (from 0 to 1). The `keypoint_visible` flag can be 0 (not labeled), 1 (labeled but not visible), or 2 (labeled and visible).

## Export to other formats

Datumaro can convert a YOLO-Pose dataset into any other format Datumaro supports. For successful conversion, the output format should support the pose estimation task (e.g., COCO, MPII, etc.).

To convert a YOLO-Pose dataset to other formats, use the following commands:

```bash
datum create
datum add -f yolo_pose <path/to/yolov8/>
datum export -f coco_person_keypoints -o <output/dir>
```
or
```bash
datum convert -if yolo_pose -i <path/to/dataset> -f coco_person_keypoints -o <path/to/dataset>
```

Alternatively, using the Python API:

```python
from datumaro.components.dataset import Dataset

data_path = 'path/to/dataset'
data_format = 'yolo_pose'

dataset = Dataset.import_from(data_path, data_format)
dataset.export('save_dir', 'coco')
```

## Export to YOLO-Pose format

Datumaro can convert an existing dataset to YOLO-Pose format if the dataset supports the pose estimation task.

Example:

```bash
datum create
datum import -f coco_person_keypoints <path/to/dataset>
datum export -f yolo_pose -o <path/to/dataset>
```

## Examples

### Example 1. Create a custom dataset in YOLO-Pose format

```python
import numpy as np
import datumaro as dm

dataset = Dataset.from_iterable(
    [
        DatasetItem(
            id=1,
            subset="train",
            media=Image(data=np.ones((8, 8, 3))),
            annotations=[
                Bbox(0, 2, 4, 2, label=0),
                Points([0.5, 0.5, 1, 0.6, 0.6, 2], label=0)
            ],
        ),
        DatasetItem(
            id=2,
            subset="valid",
            media=Image(data=np.ones((8, 8, 3))),
            annotations=[
                Bbox(0, 1, 4, 2, label=0),
                Points([0.4, 0.4, 2, 0.5, 0.5, 2], label=0)
            ],
        ),
    ],
    categories=["person"]
)

dataset.export('../yolov8_pose_dataset', format='yolo_pose')
```