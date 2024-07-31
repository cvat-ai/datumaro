import pytest

from datumaro import LabelCategories
from tests.requirements import Requirements
from tests.requirements import mark_requirement


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_add_category():
    categories = LabelCategories()
    index = categories.add("cat")
    assert index == 0
    assert len(categories) == 1
    assert categories[0].name == "cat"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_add_category_with_parent():
    categories = LabelCategories()
    index = categories.add("cat", parent="animal")
    assert index == 0
    assert len(categories) == 1
    assert categories[0].name == "cat"
    assert categories[0].parent == "animal"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_add_category_with_attributes():
    categories = LabelCategories()
    attributes = {"color", "size"}
    index = categories.add("cat", attributes=attributes)
    assert index == 0
    assert len(categories) == 1
    assert categories[0].attributes == attributes


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
@pytest.mark.parametrize("name,parent", [("cat", "animal"), ("cat", "")])
def test_add_duplicate_category(name, parent):
    categories = LabelCategories()
    categories.add(name, parent=parent)
    with pytest.raises(KeyError, match=f"Label '{parent}' '{name}' already exists"):
        categories.add(name, parent=parent)


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_potential_collision():
    """
    Previously indices were computed as (parent or "") + name

    See https://github.com/cvat-ai/datumaro/pull/51
    """
    categories = LabelCategories()
    categories.add("22", parent="parent")
    categories.add("2", parent="parent2")
    assert categories.items[0].name == "22"
    assert categories.items[0].parent == "parent"
    assert categories.items[1].name == "2"
    assert categories.items[1].parent == "parent2"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_find_category():
    categories = LabelCategories()
    categories.add("cat")
    index, category = categories.find("cat")
    assert index == 0
    assert category.name == "cat"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_find_non_existent_category():
    categories = LabelCategories()
    index, category = categories.find("dog")
    assert index is None
    assert category is None


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_from_iterable():
    categories = LabelCategories.from_iterable(["cat", "dog"])
    assert len(categories) == 2
    assert categories[0].name == "cat"
    assert categories[1].name == "dog"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_from_iterable_with_parents():
    categories = LabelCategories.from_iterable([("cat", "animal"), ("dog", "animal")])
    assert len(categories) == 2
    assert categories[0].name == "cat"
    assert categories[0].parent == "animal"
    assert categories[1].name == "dog"
    assert categories[1].parent == "animal"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_from_iterable_with_attributes():
    categories = LabelCategories.from_iterable([("cat", "animal", ["color", "size"])])
    assert len(categories) == 1
    assert categories[0].name == "cat"
    assert categories[0].parent == "animal"
    assert categories[0].attributes == {"color", "size"}


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_reindex():
    categories = LabelCategories()
    categories.add("cat")
    categories.add("dog", parent="animal")
    assert len(categories._indices) == 2


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_labels_property():
    categories = LabelCategories()
    categories.add("cat")
    categories.add("dog", parent="animal")
    labels = categories.labels
    assert labels[0] == "cat"
    assert labels[1] == "animaldog"


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_len():
    categories = LabelCategories()
    assert len(categories) == 0
    categories.add("cat")
    assert len(categories) == 1


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_contains():
    categories = LabelCategories()
    categories.add("cat")
    assert "cat" in categories
    assert "dog" not in categories


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_iter():
    categories = LabelCategories()
    categories.add("cat")
    categories.add("dog")
    items = list(categories)
    assert len(items) == 2
    assert items[0].name == "cat"
    assert items[1].name == "dog"
