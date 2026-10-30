from pathlib import Path

import pytest

from collectra import ArtefactNode, Text
from collectra.display import DisplayContext


def test_file_reference_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("texts").mkdir()
    Path("texts/note.txt").write_text("Hello", encoding="utf-8")
    text = Text(name="note", path="texts/note.txt")
    assert text() == "Hello"
    record = text.serialize()
    assert record["path"] == "texts/note.txt"
    assert "data" not in record
    assert (
        Text(name="note", **{k: v for k, v in record.items() if k != "type"})()
        == "Hello"
    )


def test_inline_and_conflicting_sources(tmp_path):
    assert Text(name="note").serialize()["data"] == ""
    assert "path" not in Text(name="note", data="Hello").serialize()
    with pytest.raises(ValueError, match="not both"):
        Text(name="note", data="", path=tmp_path / "note.txt")
    with pytest.raises(FileNotFoundError):
        Text(name="note", path=tmp_path / "missing.txt")


def test_results_loader_preserves_path(tmp_path):
    (tmp_path / "note.txt").write_text("Hello", encoding="utf-8")
    (tmp_path / "results.yaml").write_text(
        "note:\n  type: collectra.Text\n  id: note\n  path: note.txt\n"
    )
    node = ArtefactNode(name="note")
    node.add_type(Text)
    items = ArtefactNode.batch_process(tmp_path, [node])
    assert len(items) == 1
    assert items[0]() == "Hello"
    assert items[0].serialize()["path"] == "note.txt"


def test_display_resolves_explicit_path(tmp_path):
    (tmp_path / "note.md").write_text("# Hello", encoding="utf-8")
    context = DisplayContext(
        {"note": {"type": "collectra.Text", "id": "note", "path": "note.md"}},
        tmp_path,
    )
    item = context.artefact("note")
    assert item.display(context)["text"] == "# Hello"
    assert item.display(context)["format"] == "markdown"


@pytest.mark.parametrize(
    "value, expected",
    [(1, "1"), (13, "13"), (0, "0"), (1.5, "1.5"), (True, "True")],
)
def test_numbers_and_flags_from_yaml_become_text(value, expected):
    """``book: 1`` in a results.yaml is an int to YAML but text to Collectra."""
    text = Text(name="book", data=value)
    assert text.data == expected
    assert text() == expected
    assert text.serialize()["data"] == expected


def test_numeric_text_loads_the_way_the_results_loader_builds_it():
    item = {"id": "book", "data": 1}
    assert Text(name="book", **item)() == "1"
