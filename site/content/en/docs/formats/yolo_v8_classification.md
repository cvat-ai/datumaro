---
title: 'YOLOv8Classification'
linkTitle: 'YOLOv8Classification'
description: ''
---

## Format specification
Dataset format specification can be found
[here](https://docs.ultralytics.com/datasets/classify/)

Supported types of annotations:
- `Label`

Format doesn't support any attributes for annotations objects.


## Import YOLOv8 classification dataset

A Datumaro project with a ImageNet dataset can be created
in the following way:

```
datum create
datum import -f yolov8_classification <path_to_dataset>
```

Load YOLOOv8 Classification dataset through the Python API:

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset>', format='yolov8_classification')
```

For successful importing of YOLOv8 Classification dataset the input directory with dataset
should has the following structure:

```bash
dataset/
├── train
│    ├── labels.json  # optional datumaro extension. Contains original ids and labels
│    ├── label_0
│    │      ├── <image_name_0>.jpg
│    │      ├── <image_name_1>.jpg
│    │      ├── <image_name_2>.jpg
│    │      ├── ...
│    ├── label_1
│    │      ├── <image_name_0>.jpg
│    │      ├── <image_name_1>.jpg
│    │      ├── <image_name_2>.jpg
│    │      ├── ...
├── ...
```

## Export YOLOv8 Classification dataset

Datumaro can convert the dataset into any other format
[Datumaro supports](/docs/user-manual/supported_formats).
To get the expected result, convert the dataset to a format
that supports `Label` annotation objects.

```
# Using `convert` command
datum convert -if yolov8_classification -i <path_to_dataset> \
    -f voc -o <output_dir> -- --save-media

# Using Datumaro project
datum create
datum import -f yolov8_classification <path_to_dataset>
datum export -f open_images -o <output_dir>
```

And also you can convert your YOLOv8 Classification dataset using Python API

```python
import datumaro as dm

dataset = dm.Dataset.import_from('<path_to_dataset', format='yolov8_classification')

dataset.export('<output_dir>', format='vgg_face2', save_media=True)
```

> Note: some formats have extra export options. For particular format see the
> [docs](/docs/formats/) to get information about it.

## Export dataset to the YOLOv8 Classification format

If your dataset contains `Label` for images and you want to convert this
dataset into the YOLOv8 Classification format, you can use Datumaro for it:

```
# Using convert command
datum convert -if open_images -i <path_to_oid> \
    -f yolov8_classification -o <output_dir> -- --save-media --save-dataset-meta

# Using Datumaro project
datum create
datum import -f open_images <path_to_oid>
datum export -f yolov8_classification -o <output_dir>
```

Extra options for exporting to YOLOv8 Classification formats:
- `--save-media` allow to export dataset with saving media files
  (by default `False`)
- `--save-dataset-meta` - allow to export dataset with saving dataset meta
  file (by default `False`)
