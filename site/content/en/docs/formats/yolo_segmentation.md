---
title: 'YOLO-Segmentation'
linkTitle: 'YOLO-Segmentation'
description: ''
---

## Format specification

The YOLO-Segmentation dataset format is designed for training and validating object segmentation models using the YOLO framework. Detailed specifications for this format can be found in the official [YOLO Segmentation documentation](https://docs.ultralytics.com/datasets/segment/).

Supported annotation types:
- `Polygons`

The format supports various subset names, except classes, names, and backup.

> Note, that by default, the YOLO framework does not expect any subset names,
  except `train` and `valid`, Datumaro supports this as an extension.
  If there is no subset separation in a project, the data
  will be saved in the `train` subset.

## Import YOLO-Segmentation dataset

To create a Datumaro project with a YOLO-Segmentation source, use the following commands:

```bash
datum create
datum import --format yolo_segmentation <path/to/dataset>
```

The YOLO-Segmentation dataset directory should have the following structure:

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

`data.yaml` should have the following content. It is not necessary to have both subsets, but necessary to have one of them. Additionally, class numbers should be zero-indexed (start with 0):

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path:  ./ # dataset root dir
train: train.txt  # train images (relative to 'path')
val: val.txt  # val images (relative to 'path')
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
# labels/train/image1.txt:
# <label_index> <x1> <y1> <x2> <y2> ... <xn> <yn>
0 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
...
```

Each row in these label files contains the following information about the object instance:
- Object class index: An integer representing the class of the object (e.g., 0 for person, 2 for car, etc.).
- Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.

> Note : The length of each row does not have to be equal
    Each segmentation label must have a minimum of 3 xy points: <class-index> <x1> <y1> <x2> <y2> <x3> <y3>


## Export to other formats

Datumaro can convert a YOLO-Segmentation dataset into any other format Datumaro supports. For successful conversion, the output format should support the object segmentation task (e.g., COCO, Pascal VOC, etc.).

To convert a YOLO-Segmentation dataset to other formats, use the following commands:

```bash
datum create
datum add -f yolo_segmentation <path/to/yolo_segmentation/>
datum export -f coco_stuff -o <output/dir>
```
or
```bash
datum convert -if yolo_segmentation -i <path/to/dataset> -f coco_stuff -o <path/to/dataset>
```

Alternatively, using the Python API:

```python
from datumaro.components.dataset import Dataset

data_path = 'path/to/dataset'
data_format = 'yolo_segmentation'

dataset = Dataset.import_from(data_path, data_format)
dataset.export('save_dir', 'coco_stuff')
```

## Export to YOLO-Segmentation format

Datumaro can convert an existing dataset to YOLO-Segmentation format if the dataset supports the object segmentation task.

Example:

```bash
datum create
datum import -f coco_stuff <path/to/dataset>
datum export -f yolo_segmentation -o <path/to/dataset>
```

## Examples

### Example 1. Create a custom dataset in YOLO-Segmentation format

```python
import numpy as np
import datumaro as dm
from datumaro.components.annotation import Polygon

dataset = dm.Dataset.from_iterable(
    [
        dm.DatasetItem(
            id="image_001",
            subset="train",
            image=np.ones((20, 20, 3)),
            annotations=[
                Polygon([3.0, 1.0, 8.0, 5.0, 1.0, 1.0], label=1),
                Polygon([4.0, 4.0, 4.0, 8.0, 8.0, 4.0], label=2),
            ],
        ),
        dm.DatasetItem(
            id="image_002",
            subset="train",
            image=np.ones((15, 10, 3)),
            annotations=[
                Polygon([1.0, 1.0, 1.0, 4.0, 4.0, 1.0], label=3)
            ],
        ),
    ],
    categories=["label_" + str(i) for i in range(4)],
)

dataset.export('../yolo_segmentation_dataset', format='yolo_segmentation')
```