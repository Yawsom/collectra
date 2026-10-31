"""Artefacts are pydantic models: typed fields with class-level defaults."""

import copy
from pathlib import Path

import pytest
from PIL import Image as PilImage
from pydantic import BaseModel, ValidationError

from collectra.types.images import Image, ImageCrop, ImageSegmentation, Orientation
from collectra.types.links import Link
from collectra.types.texts import Text


@pytest.fixture
def folder(tmp_path, monkeypatch):
    PilImage.new("RGB", (40, 20), (255, 0, 0)).save(tmp_path / "page.png")
    mask = PilImage.new("L", (40, 20), 0)
    mask.paste(1, (10, 5, 30, 15))
    mask.save(tmp_path / "mask.png")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def crop(**kwargs):
    defaults = dict(name="human", id="human-1", data="page.png")
    return ImageCrop(**{**defaults, **kwargs})


def test_every_artefact_class_is_a_pydantic_model():
    for cls in (Image, ImageCrop, ImageSegmentation, Text, Link):
        assert issubclass(cls, BaseModel), cls


def test_defaults_are_declared_as_class_level_fields():
    assert Text.model_fields["id"].default == ""
    assert Text.model_fields["orientation"].default == "north"
    assert Text.model_fields["data"].default is None
    assert ImageCrop.model_fields["x_center"].default == 0.5
    assert ImageCrop.model_fields["width_relative"].default == 1.0
    assert ImageCrop.model_fields["confidence"].default is None
    assert Image.model_fields["orientation"].default == Orientation.NORTH


def test_values_are_validated_and_coerced(folder):
    item = crop(x_center="0.25", confidence=1)
    assert item.x_center == 0.25
    assert item.confidence == 1.0
    with pytest.raises(ValidationError):
        crop(x_center="left")


def test_assignment_is_validated(folder):
    item = crop()
    item.x_center = "0.3"
    assert item.x_center == 0.3
    with pytest.raises(ValidationError):
        item.x_center = "left"
    assert item.x_center == 0.3


def test_orientation_accepts_text_and_is_an_enum_on_images(folder):
    assert crop(orientation="east").orientation is Orientation.EAST
    assert Text(name="t", orientation="east").orientation == "east"
    item = crop()
    item.orientation = "south"
    assert item.orientation is Orientation.SOUTH
    with pytest.raises(ValueError):
        crop(orientation="sideways")


def test_undeclared_attributes_are_accepted():
    text = Text(name="t", legacy_option=3)
    assert text.legacy_option == 3
    text.note = "x"
    assert text.note == "x"


def test_parents_always_hold_a_list_of_ids():
    assert Text(name="t", parents="a").parents == ["a"]
    assert Text(name="t", parents=["a", "b"]).parents == ["a", "b"]
    assert Text(name="t").parents == []


def test_automatic_ids_are_marked_provisional_and_explicit_ones_are_not():
    assert Text(name="t")._auto_id is True
    assert Text(name="t", id="chosen")._auto_id is False
    kept = Text(name="t")
    kept.parents = ["x"]  # assigning a field must not lose the marker
    assert kept._auto_id is True


def test_source_file_is_not_part_of_the_record(folder):
    text = Text(name="t", id="t", data="x")
    text.source_file = folder / "results.yaml"
    assert "source_file" not in repr(text)
    assert "source_file" not in text.model_dump()
    assert text == Text(name="t", id="t", data="x")


def test_equality_compares_values(folder):
    assert Text(name="a", id="i", data="x") == Text(name="a", id="i", data="x")
    assert Text(name="a", id="i", data="x") != Text(name="a", id="i", data="y")
    assert Text(name="a", data="x") != Text(name="a", id="zz", data="x")
    assert Text(name="a", id="i") != crop()
    assert crop() == crop()
    assert crop() != crop(x_center=0.1)


def test_deep_copies_are_equal_and_independent(folder):
    item = crop(parents=["page"])
    clone = copy.deepcopy(item)
    assert clone == item
    clone.parents.append("other")
    assert item.parents == ["page"]


# --- what gets written to results.yaml must not change --------------------


def test_serialised_records_are_unchanged(folder):
    page = Image(name="page", id="page", data="page.png")
    assert page.serialize() == {
        "type": "collectra.Image",
        "id": "page",
        "embeddings": [],
        "orientation": "north",
        "ensemble": [],
        "data": "page.png",
    }
    box = crop(
        x_center=0.25,
        y_center=0.5,
        width_relative=0.5,
        height_relative=0.5,
        confidence=0.9,
        parents=["page"],
        partition="train",
        orientation="east",
    )
    assert box.serialize() == {
        "type": "collectra.ImageCrop",
        "id": "human-1",
        "parents": "page",
        "embeddings": [],
        "orientation": "east",
        "ensemble": [],
        "data": "page.png",
        "x_center": 0.25,
        "y_center": 0.5,
        "width_relative": 0.5,
        "height_relative": 0.5,
        "confidence": 0.9,
    }
    region = ImageSegmentation(
        name="region", id="region", data="page.png", mask="mask.png", parents=["page"]
    )
    assert region.serialize() == {
        "type": "collectra.ImageSegmentation",
        "id": "region",
        "parents": "page",
        "embeddings": [],
        "orientation": "north",
        "ensemble": [],
        "data": "page.png",
        "mask": "mask.png",
    }
    note = Text(name="note", id="note", data="hello", parents=["a", "b"])
    assert note.serialize() == {
        "type": "collectra.Text",
        "id": "note",
        "parents": ["a", "b"],
        "embeddings": [],
        "orientation": "north",
        "ensemble": [],
        "data": "hello",
    }
    assert Link(name="l", id="human_link", parents=["human-1"]).serialize() == {
        "type": "collectra.Link",
        "id": "human_link",
        "parents": "human-1",
    }


def test_attribute_listing_keeps_only_the_recorded_fields(folder):
    assert list(crop().attributes) == [
        "id", "parents", "embeddings", "orientation", "ensemble", "data",
        "x_center", "y_center", "width_relative", "height_relative", "confidence",
    ]  # fmt: skip


# --- links present themselves as the artefact they point to ---------------


def test_links_pass_isinstance_checks_for_their_bound_target(folder):
    link = Link(name="l", id="l", parents=["human-1"])
    assert isinstance(link, Link)
    assert not isinstance(link, ImageCrop)
    target = crop()
    link.bind({"human-1": target})
    assert isinstance(link, Link)
    assert isinstance(link, ImageCrop)
    assert isinstance(link, Image)
    assert not isinstance(link, Text)
    assert type(link) is Link
    assert link.x_center == 0.5
    assert link.resolve() is target


def test_link_chains_resolve_to_the_final_artefact(folder):
    inner = Link(name="inner", id="inner", parents=["human-1"])
    outer = Link(name="outer", id="outer", parents=["inner"])
    target = crop()
    inner.bind({"human-1": target})
    outer.bind({"inner": inner})
    assert outer.resolve() is target
    assert isinstance(outer, ImageCrop)


def test_unbound_links_do_not_expose_target_attributes():
    link = Link(name="l", id="l", parents=["x"])
    with pytest.raises(AttributeError):
        link.x_center
    with pytest.raises(RuntimeError, match="unresolved"):
        link.resolve()


def test_links_compare_without_their_runtime_target(folder):
    first = Link(name="l", id="l", parents=["human-1"])
    second = Link(name="l", id="l", parents=["human-1"])
    first.bind({"human-1": crop()})
    assert first == second


def test_fields_can_still_be_given_positionally_in_declaration_order():
    text = Text("empty", "an-id")
    assert (text.name, text.id) == ("empty", "an-id")
    assert Text("t", "i", ["p"], "train").partition == "train"
    with pytest.raises(TypeError, match="multiple values"):
        Text("t", name="again")
    with pytest.raises(TypeError, match="positional"):
        Text(*["x"] * 30)


def test_runtime_fields_cannot_be_given_positionally(folder):
    # raw_width and raw_height are measured from the file, not passed in.
    item = Image("page", "page", [], "", [], "north", [], "page.png")
    assert item.data == folder / "page.png"
    assert (item.raw_width, item.raw_height) == (40, 20)
