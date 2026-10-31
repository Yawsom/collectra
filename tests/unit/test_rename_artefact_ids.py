"""scripts/rename_artefact_ids.py gives artefacts ids named after their node."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image as PilImage

SCRIPT = Path(__file__).parents[2] / "scripts" / "rename_artefact_ids.py"
spec = importlib.util.spec_from_file_location("rename_artefact_ids", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

META = {"workflow": "t", "version": "1", "timestamp": "t"}


def text(id, data="x", **extra):
    return {"type": "collectra.Text", "id": id, "data": data, **extra}


def write(folder: Path, nodes: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {"collectra_results_metadata": META}
    for name, entries in nodes.items():
        data[name] = entries
    (folder / "results.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def load(folder: Path) -> dict:
    return yaml.safe_load((folder / "results.yaml").read_text())


def ids(folder: Path, key: str) -> list[str]:
    value = load(folder)[key]
    return [e["id"] for e in (value if isinstance(value, list) else [value])]


def run(args, capsys):
    code = script.main([str(a) for a in args])
    return code, capsys.readouterr().out


# --- the naming rule ------------------------------------------------------------


def test_a_lone_artefact_is_named_after_its_node(tmp_path):
    folder = write(tmp_path / "p", {"tei": text("tei-3f2a-uuid")})
    script.main([str(folder)])
    assert ids(folder, "tei") == ["tei"]


def test_a_lone_artefact_in_a_list_is_named_after_its_node_too(tmp_path):
    folder = write(tmp_path / "p", {"tei": [text("old")]})
    script.main([str(folder)])
    assert ids(folder, "tei") == ["tei"]


def test_several_artefacts_are_numbered_from_one_in_order(tmp_path):
    folder = write(
        tmp_path / "p", {"parts": [text("zz", "a"), text("aa", "b"), text("mm", "c")]}
    )
    script.main([str(folder)])
    value = load(folder)["parts"]
    assert [e["id"] for e in value] == ["parts-1", "parts-2", "parts-3"]
    assert [e["data"] for e in value] == ["a", "b", "c"]  # order is kept


def test_new_ids_are_planned_from_the_nodes_alone():
    plan = script.plan_ids({"a": [text("x")], "b": [text("y"), text("z")], "c": []})
    assert plan == {"x": "a", "y": "b-1", "z": "b-2"}


# --- references are rewritten -------------------------------------------------------


def test_parents_follow_the_renamed_ids_whether_single_or_listed(tmp_path):
    folder = write(
        tmp_path / "p",
        {
            "boxes": [text("b-old-1", "1"), text("b-old-2", "2")],
            "words": [text("w-old", "w", parents="b-old-2")],
            "summary": text("s-old", "s", parents=["b-old-1", "w-old"]),
        },
    )
    script.main([str(folder)])
    data = load(folder)
    assert data["words"][0]["parents"] == "boxes-2"
    assert data["summary"]["parents"] == ["boxes-1", "words"]


def test_references_to_ids_outside_the_file_are_left_alone(tmp_path):
    folder = write(
        tmp_path / "p",
        {
            "boxes": [text("old-1"), text("old-2")],
            "ensembled": text("e", parents=["/other/results::far-away", "old-1"]),
        },
    )
    script.main([str(folder)])
    assert load(folder)["ensembled"]["parents"] == [
        "/other/results::far-away",
        "boxes-1",
    ]


def test_a_new_id_that_equals_another_old_id_is_not_confused(tmp_path):
    # 'b-2' is the old id of the first item and the new id of the second.
    folder = write(
        tmp_path / "p",
        {
            "b": [text("b-2", "first"), text("b-1", "second")],
            "child": text("c", parents="b-2"),
        },
    )
    script.main([str(folder)])
    assert [e["data"] for e in load(folder)["b"]] == ["first", "second"]
    assert load(folder)["child"]["parents"] == "b-1"  # the first item, now b-1


def test_ensemble_references_are_rewritten(tmp_path):
    folder = write(
        tmp_path / "p",
        {
            "boxes": [text("o1"), text("o2")],
            "x": text("x", ensemble=["o2", "elsewhere"]),
        },
    )
    script.main([str(folder)])
    assert load(folder)["x"]["ensemble"] == ["boxes-2", "elsewhere"]


def test_files_named_by_the_old_ids_are_not_touched(tmp_path):
    folder = write(
        tmp_path / "p",
        {"tei": {"type": "collectra.Text", "id": "tei-old", "path": "tei/tei-old.xml"}},
    )
    (folder / "tei").mkdir()
    (folder / "tei" / "tei-old.xml").write_text("<x/>")
    script.main([str(folder)])
    assert load(folder)["tei"]["path"] == "tei/tei-old.xml"
    assert (folder / "tei" / "tei-old.xml").read_text() == "<x/>"


# --- what is left alone ------------------------------------------------------------------


def test_metadata_empty_nodes_and_other_content_are_preserved(tmp_path):
    folder = write(
        tmp_path / "p",
        {"empty": [], "note": text("n-old", "keep me"), "plain": "just a string"},
    )
    script.main([str(folder)])
    data = load(folder)
    assert data["collectra_results_metadata"] == META
    assert data["empty"] == []
    assert data["note"]["data"] == "keep me"
    assert data["plain"] == "just a string"


def test_entries_that_are_not_artefact_records_are_ignored_but_missing_ids_are_filled(
    tmp_path,
):
    folder = write(
        tmp_path / "p",
        {"mixed": [{"type": "collectra.Text", "data": "no id"}, "stray", text("has")]},
    )
    script.main([str(folder)])
    value = load(folder)["mixed"]
    assert [e["id"] for e in value if isinstance(e, dict)] == ["mixed-1", "mixed-2"]
    assert "stray" in value


def test_files_already_in_the_right_form_are_not_rewritten(tmp_path, capsys):
    folder = write(
        tmp_path / "p",
        {"tei": text("tei"), "parts": [text("parts-1"), text("parts-2")]},
    )
    before = (folder / "results.yaml").read_text()
    code, out = run([folder], capsys)
    assert (folder / "results.yaml").read_text() == before
    assert code == 0
    assert "Nothing to rename" in out


def test_running_twice_changes_nothing_the_second_time(tmp_path):
    folder = write(
        tmp_path / "p",
        {"parts": [text("a"), text("b")], "child": text("c", parents="b")},
    )
    script.main([str(folder)])
    once = (folder / "results.yaml").read_text()
    script.main([str(folder)])
    assert (folder / "results.yaml").read_text() == once


# --- options ------------------------------------------------------------------------------


def test_dry_run_reports_and_writes_nothing(tmp_path, capsys):
    folder = write(tmp_path / "p", {"parts": [text("a"), text("b")]})
    before = (folder / "results.yaml").read_text()
    code, out = run([folder, "--dry-run"], capsys)
    assert (folder / "results.yaml").read_text() == before
    assert "Would rename" in out and "parts-1" in out
    assert code == 0


def test_a_summary_says_how_many_were_renamed(tmp_path, capsys):
    folder = write(tmp_path / "p", {"parts": [text("a"), text("b")], "tei": text("t")})
    code, out = run([folder], capsys)
    assert "Renamed 3" in out
    assert code == 0


def test_node_restricts_which_nodes_are_renamed_but_references_still_follow(tmp_path):
    folder = write(
        tmp_path / "p",
        {
            "parts": [text("a"), text("b")],
            "other": [text("c"), text("d")],
            "child": text("e", parents=["a", "c"]),
        },
    )
    script.main([str(folder), "--node", "parts"])
    assert ids(folder, "parts") == ["parts-1", "parts-2"]
    assert ids(folder, "other") == ["c", "d"]  # not selected
    assert load(folder)["child"]["parents"] == ["parts-1", "c"]


def test_json_lists_every_rename(tmp_path, capsys):
    folder = write(tmp_path / "p", {"parts": [text("a"), text("b")], "tei": text("t")})
    code, out = run([folder, "--json"], capsys)
    (record,) = json.loads(out)
    assert record["file"].endswith("results.yaml")
    assert record["renamed"] == [
        {"node": "parts", "old": "a", "new": "parts-1"},
        {"node": "parts", "old": "b", "new": "parts-2"},
        {"node": "tei", "old": "t", "new": "tei"},
    ]
    assert record["applied"] is True


# --- many files and awkward ones ---------------------------------------------------------


def test_folders_and_parents_are_searched(tmp_path, capsys):
    a = write(tmp_path / "set" / "a.livy", {"n": [text("x"), text("y")]})
    b = write(tmp_path / "set" / "deep" / "b.livy", {"n": text("z")})
    code, out = run([tmp_path / "set"], capsys)
    assert ids(a, "n") == ["n-1", "n-2"] and ids(b, "n") == ["n"]
    assert "2 results file" in out
    assert run([tmp_path / "nothing"], capsys)[0] == 2


def test_a_file_with_ids_that_would_collide_is_skipped_and_reported(tmp_path, capsys):
    # Node 'a' with two items becomes a-1, a-2; node 'a-1' would also be 'a-1'.
    folder = write(tmp_path / "p", {"a": [text("x"), text("y")], "a-1": text("z")})
    before = (folder / "results.yaml").read_text()
    code, out = run([folder], capsys)
    assert (folder / "results.yaml").read_text() == before
    assert code == 1
    assert "collide" in out


def test_ids_used_by_two_nodes_make_references_ambiguous_so_the_file_is_skipped(
    tmp_path, capsys
):
    folder = write(
        tmp_path / "p",
        {"a": text("same"), "b": text("same"), "c": text("c", parents="same")},
    )
    before = (folder / "results.yaml").read_text()
    code, out = run([folder], capsys)
    assert (folder / "results.yaml").read_text() == before
    assert code == 1
    assert "more than one" in out


def test_other_files_still_get_renamed_when_one_is_skipped(tmp_path, capsys):
    bad = write(tmp_path / "bad", {"a": text("same"), "b": text("same")})
    good = write(tmp_path / "good", {"n": [text("x"), text("y")]})
    code, out = run([tmp_path], capsys)
    assert ids(good, "n") == ["n-1", "n-2"]
    assert code == 1


# --- the result is a valid collectra file ------------------------------------------------


def test_the_renamed_file_loads_with_collectra_and_lineage_intact(tmp_path):
    from collectra.types.base import ArtefactNode
    from collectra.types.images import ImageCrop
    from collectra.types.texts import Text

    folder = tmp_path / "p"
    folder.mkdir()
    PilImage.new("RGB", (20, 10)).save(folder / "page.png")
    crop = lambda id, x: {
        "type": "collectra.ImageCrop",
        "id": id,
        "data": "page.png",
        "x_center": x,
        "y_center": 0.5,
        "width_relative": 0.2,
        "height_relative": 0.5,
    }
    write(
        folder,
        {
            "boxes": [crop("box-uuid-1", 0.2), crop("box-uuid-2", 0.7)],
            "words": text("w-uuid", "read", parents="box-uuid-2"),
        },
    )
    script.main([str(folder)])
    boxes = ArtefactNode("boxes", types={ImageCrop})
    boxes.process("boxes", folder)
    words = ArtefactNode("words", types={Text})
    words.process("words", folder)
    assert sorted(boxes.items) == ["boxes-1", "boxes-2"]
    (word,) = words.items.values()
    assert word.id == "words"
    assert word.parents == ["boxes-2"] and "boxes-2" in boxes.items
