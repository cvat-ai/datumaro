from datumaro.components.dataset_base import DatasetItem, SubsetBase


class MyBase(SubsetBase):
    def __iter__(self):
        yield from [
            DatasetItem("1"),
            DatasetItem("2"),
        ]
