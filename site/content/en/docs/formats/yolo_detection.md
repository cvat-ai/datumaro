---
title: 'YOLO-Detection'
linkTitle: 'YOLO-Detection'
description: ''
---

## Format specification

The YOLO-Detection dataset format is designed for training and validating object detection models using the YOLO framework. Detailed specifications for this format can be found in the official [YOLO Detection documentation](https://docs.ultralytics.com/datasets/detect/).

Supported annotation types:
- `Bounding boxes`

The format supports various subset names, except classes, names, and backup.

The format supports arbitrary subset names, except `classes`, `names` and `backup`.

> Note, that by default, the YOLO framework does not expect any subset names,
  except `train` and `valid`, Datumaro supports this as an extension.
  If there is no subset separation in a project, the data
  will be saved in the `train` subset.

## Import YOLO-Detection dataset
To create a Datumaro project with a YOLO-Detection source, use the following commands:

```bash
datum create
datum import --format yolo_detection <path/to/dataset>
```

The YOLO-Detection dataset directory should have the following structure:

```bash
└─ yolo_segmentation_dataset/
   │ # a list of non-format labels (optional)  # file with list of classes
   ├── data.yaml    # file with dataset information
   ├── train.txt    # list of image paths in train subset [Optional]
   ├── valid.txt    # list of image paths in valid subset [Optional]
   │
   ├── images/
   │   ├── train/  # directory with images for train subset
   │   │    ├── image1.jpg
   │   │    ├── image2.jpg
   │   │    ├── image3.jpg
   │   │    └── ...
   │   ├── valid/  # directory with images for validation subset
   │   │    ├── image11.jpg
   │   │    ├── image12.jpg
   │   │    ├── image13.jpg
   │   │    └── ...
   ├── labels/
   │   ├── train/  # directory with annotations for train subset
   │   │    ├── image1.txt
   │   │    ├── image2.txt
   │   │    ├── image3.txt
   │   │    └── ...
   │   ├── valid/  # directory with annotations for validation subset
   │   │    ├── image11.txt
   │   │    ├── image12.txt
   │   │    ├── image13.txt
   │   │    └── ...
```

`data.yaml` should have the following content, it is not necessary to have both subsets, but necessary to have one of them. Additionally, class numbers should be zero-indexed (start with 0).:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path:  ./ # dataset root dir
train: train.txt  # train images (relative to 'path') 4 images
val: val.txt  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  # ...
  77: teddy bear
  78: hair drier
  79: toothbrush
```

Files `train.txt` and `valid.txt` should have the following structure:

```txt
<path/to/image1.jpg>
<path/to/image2.jpg>
...
```

Files in directories `labels/train/` and `labels/valid/` should contain information about labeled bounding boxes for images in `images/train` and `images/valid` respectively. If there are no objects in an image, no `.txt` file is required:

```txt
# labels/image1.txt:
# <label_index> <x_center> <y_center> <width> <height>
0 0.250000 0.400000 0.300000 0.400000
3 0.600000 0.400000 0.400000 0.266667
...
```

Box coordinates must be in normalized `xywh` format (from 0 to 1). If your boxes are in pixels, you should divide `x_center` and `width` by `image width`, and `y_center` and `height` by `image height`

## Export to other formats

Datumaro can convert a YOLO-Detection dataset into any other format Datumaro supports. For successful conversion, the output format should support the object detection task (e.g., Pascal VOC, COCO, TF Detection API, etc.).

To convert a YOLO-Detection dataset to other formats, use the following commands:

```bash
datum create
datum add -f yolo_detection <path/to/yolov8/>
datum export -f coco_instances -o <output/dir>
```
or
```bash
datum convert -if yolo_detection -i <path/to/dataset> -f coco_instances -o <path/to/dataset>
```

Alternatively, using the Python API:

```python
from datumaro.components.dataset import Dataset

data_path = 'path/to/dataset'
data_format = 'yolo_detection'

dataset = Dataset.import_from(data_path, data_format)
dataset.export('save_dir', 'coco_instances')
```

## Export to YOLO-Detection format
Datumaro can convert an existing dataset to YOLO-Detection format if the dataset supports the object detection task.

Example:

```bash
datum create
datum import -f coco_instances <path/to/dataset>
datum export -f yolo_detection -o <path/to/dataset>
```

## Examples

### Example 1. Create a custom dataset in YOLO-Detection format

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
                Bbox(0, 2, 4, 2, label=2),
                Bbox(0, 1, 2, 3, label=4),
            ],
        ),
        DatasetItem(
            id=2,
            subset="valid",
            media=Image(data=np.ones((8, 8, 3))),
            annotations=[
                Bbox(0, 1, 5, 2, label=2),
                Bbox(0, 2, 3, 2, label=5),
                Bbox(0, 2, 4, 2, label=6),
                Bbox(0, 7, 3, 2, label=7),
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(10)],
)

dataset.export('../yolov8_dataset', format='yolo_detection')
```