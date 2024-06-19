---
title: 'YOLO-OrientedBox'
linkTitle: 'YOLO-OrientedBox'
description: ''
---

## Format specification

The YOLO-OrientedBox dataset format is designed for training and validating object detection models that use oriented bounding boxes (OBB). This format is especially useful for detecting objects in aerial and satellite imagery, where objects may not be aligned with the image axes. Detailed specifications for this format can be found in the official [YOLO Oriented Bounding Box documentation](https://docs.ultralytics.com/datasets/obb/).

Supported annotation types:
- `Oriented Bounding Boxes`

The format supports various subset names, except classes, names, and backup.

> Note, that by default, the YOLO framework does not expect any subset names, except `train` and `valid`. Datumaro supports this as an extension. If there is no subset separation in a project, the data will be saved in the `train` subset.

## Import YOLO-OrientedBox dataset

To create a Datumaro project with a YOLO-OrientedBox source, use the following commands:

```bash
datum create
datum import --format yolo_orientedbox <path/to/dataset>
```

The YOLO-OrientedBox dataset directory should have the following structure:

```bash
└─ yolo_orientedbox_dataset/
   ├── data.yaml    # file with dataset information
   ├── train.txt    # list of image paths in train subset [Optional]
   ├── valid.txt    # list of image paths in valid subset [Optional]
   ├── images/
   │   ├── train/   # directory with images for train subset
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── valid/   # directory with images for valid subset
   │   │   ├── image11.jpg
   │   │   └── ...
   ├── labels/
   │   ├── train/   # directory with annotations for train subset
   │   │   ├── image1.txt
   │   │   ├── image2.txt
   │   │   └── ...
   │   ├── valid/   # directory with annotations for valid subset
   │   │   ├── image11.txt
   │   │   └── ...
```

`data.yaml` should have the following content. It is not necessary to have both subsets, but necessary to have one of them. Additionally, class numbers should be zero-indexed (start with 0):

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path:  ./  # dataset root dir
train: train.txt  # train images (relative to 'path')
val: val.txt  # val images (relative to 'path')
test:  # test images (optional)

# Classes
names:
  0: airplane
  1: ship
  2: storage tank
  3: baseball diamond
  # ...
  14: helicopter
```

Files `train.txt` and `valid.txt` should have the following structure:

```txt
<path/to/image1.jpg>
<path/to/image2.jpg>
...
```

Files in directories `labels/train/` and `labels/valid/` should contain information about labeled oriented bounding boxes for images. If there are no objects in an image, no `.txt` file is required:

```txt
# labels/image1.txt:
# class_index, x1, y1, x2, y2, x3, y3, x4, y4
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
...
```

Bounding box coordinates must be in normalized `xyxyxyxy` format (from 0 to 1). If your boxes are in pixels, you should divide `x` by `image width` and `y` by `image height`. Internally datumaro processes these oriented bounding boxes in `xywhr` format, where `r` is the angle

## Export to other formats

Datumaro can convert a YOLO-OrientedBox dataset into any other format Datumaro supports. For successful conversion, the output format should support the oriented bounding box detection task (e.g., Pascal VOC, COCO, TF Detection API, etc.).

To convert a YOLO-OrientedBox dataset to other formats, use the following commands:

```bash
datum convert -if yolo_orientedbox -i <path/to/dataset> -f coco_instances -o <path/to/dataset>
```

Alternatively, using the Python API:

```python
from datumaro.components.dataset import Dataset

data_path = 'path/to/dataset'
data_format = 'yolo_orientedbox'

dataset = Dataset.import_from(data_path, data_format)
dataset.export('save_dir', 'coco_instances')
```

## Export to YOLO-OrientedBox format

Datumaro can convert an existing dataset to YOLO-OrientedBox format if the dataset supports the oriented bounding box detection task.

Example:

```bash
datum create
datum import -f coco_instances <path/to/dataset>
datum export -f yolo_orientedbox -o <path/to/dataset>
```

## Examples

### Example 1. Create a custom dataset in YOLO-OrientedBox format

```python
import numpy as np
import datumaro as dm

dataset = Dataset.from_iterable(
    [
        DatasetItem(
            id='image1',
            subset="train",
            media=Image(data=np.ones((8, 8, 3))),
            annotations=[
                Bbox(0, 2, 4, 2, 45, label=2, attributes = {"angle": 0.34}),
                Bbox(0, 1, 2, 3, 30, label=4, attributes = {"angle": 0.60}),
            ],
        ),
        DatasetItem(
            id='image2',
            subset="valid",
            media=Image(data=np.ones((8, 8, 3))),
            annotations=[
                Bbox(0, 1, 5, 2, 15, label=2, attributes = {"angle": 0.77}),
                Bbox(0, 2, 3, 2, 60, label=5, attributes = {"angle": 0.988}),
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(10)],
)

dataset.export('../yolov8_orientedbox_dataset', format='yolo_orientedbox')
```