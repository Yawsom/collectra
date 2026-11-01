"""scripts/find_overlapping_crops.py reports overlapping ImageCrop boxes."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image as PilImage

from collectra.types.images import ImageCrop, ImageSegmentation
from collectra.types.links import Link

SCRIPT = Path(__file__).parents[2] / "scripts" / "find_overlapping_crops.py"
spec = importlib.util.spec_from_file_location("find_overlapping_crops", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def box(xc, yc, w, h):
    return dict(x_center=xc, y_center=yc, width_relative=w, height_relative=h)


def write_results(folder: Path, nodes: dict, image="page.png", size=(200, 100)):
    """A result folder with ``image`` and ``nodes`` ({node: [artefact records]})."""
    folder.mkdir(parents=True, exist_ok=True)
    PilImage.new("RGB", size, "white").save(folder / image)
    for name, entries in nodes.items():
        for entry in entries:
            entry.setdefault("data", image)
    data = {
        "collectra_results_metadata": {
            "workflow": "t",
            "version": "1",
            "timestamp": "t",
        }
    }
    for name, entries in nodes.items():
        records = [dict(e) for e in entries]
        data[name] = records if len(records) != 1 else records[0]
    (folder / "results.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def crop(id, **geometry):
    return {"type": "collectra.ImageCrop", "id": id, **box(*geometry["b"])}


# --- the geometry ----------------------------------------------------------------


def make(tmp_path, xc, yc, w, h, id="c"):
    folder = write_results(tmp_path / id, {"n": [crop(id, b=(xc, yc, w, h))]})
    (loaded,) = script.load_crops(folder / "results.yaml")[0]
    return loaded


def test_a_crop_is_a_box_in_relative_corners(tmp_path):
    assert script.relative_box(make(tmp_path, 0.5, 0.5, 0.5, 0.25)) == pytest.approx(
        (0.25, 0.375, 0.75, 0.625)
    )


def test_overlapping_boxes_report_intersection_iou_and_share_of_the_smaller(tmp_path):
    a = make(tmp_path, 0.25, 0.5, 0.5, 1.0, "a")  # left half
    b = make(tmp_path, 0.5, 0.5, 0.5, 1.0, "b")  # middle half: shares a quarter
    found = script.overlap_between(a, b)
    assert found.intersection == pytest.approx((0.25, 0.0, 0.5, 1.0))
    assert found.iou == pytest.approx(1 / 3)
    assert found.share_of_smaller == pytest.approx(0.5)


def test_a_box_inside_another_is_fully_overlapped(tmp_path):
    outer = make(tmp_path, 0.5, 0.5, 1.0, 1.0, "outer")
    inner = make(tmp_path, 0.5, 0.5, 0.2, 0.2, "inner")
    found = script.overlap_between(outer, inner)
    assert found.share_of_smaller == pytest.approx(1.0)
    assert found.iou == pytest.approx(0.04)


def test_boxes_that_only_touch_or_are_apart_do_not_overlap(tmp_path):
    left = make(tmp_path, 0.25, 0.5, 0.5, 1.0, "left")
    right = make(tmp_path, 0.75, 0.5, 0.5, 1.0, "right")  # shares only an edge
    far = make(tmp_path, 0.9, 0.1, 0.1, 0.1, "far")
    assert script.overlap_between(left, right) is None
    assert script.overlap_between(left, far) is None


# --- loading ---------------------------------------------------------------------


def test_only_image_crops_are_loaded(tmp_path):
    folder = write_results(
        tmp_path / "p",
        {
            "boxes": [
                crop("b1", b=(0.2, 0.5, 0.2, 0.5)),
                crop("b2", b=(0.6, 0.5, 0.2, 0.5)),
            ],
            "region": [
                {"type": "collectra.Text", "id": "t", "data": "words"},
            ],
            "alias": [{"type": "collectra.Link", "id": "l", "parents": "b1"}],
        },
    )
    crops, skipped = script.load_crops(folder / "results.yaml")
    assert sorted(c.id for c in crops) == ["b1", "b2"]
    assert all(type(c) is ImageCrop for c in crops)
    assert skipped == 0


def test_crops_that_cannot_be_loaded_are_counted_not_fatal(tmp_path):
    folder = write_results(
        tmp_path / "p", {"boxes": [crop("ok", b=(0.5, 0.5, 0.2, 0.2))]}
    )
    data = yaml.safe_load((folder / "results.yaml").read_text())
    data["lost"] = {
        "type": "collectra.ImageCrop",
        "id": "lost",
        "data": "missing.png",
        **box(0.5, 0.5, 0.1, 0.1),
    }
    (folder / "results.yaml").write_text(yaml.safe_dump(data))
    crops, skipped = script.load_crops(folder / "results.yaml")
    assert [c.id for c in crops] == ["ok"]
    assert skipped == 1


def test_results_files_are_found_from_files_folders_and_parents(tmp_path):
    one = write_results(
        tmp_path / "set" / "a.livy", {"n": [crop("x", b=(0.5, 0.5, 0.1, 0.1))]}
    )
    two = write_results(
        tmp_path / "set" / "deeper" / "b.livy",
        {"n": [crop("y", b=(0.5, 0.5, 0.1, 0.1))]},
    )
    assert script.find_results_files([one / "results.yaml"]) == [one / "results.yaml"]
    assert script.find_results_files([one]) == [one / "results.yaml"]
    assert script.find_results_files([tmp_path / "set"]) == [
        one / "results.yaml",
        two / "results.yaml",
    ]
    assert script.find_results_files([tmp_path / "nothing"]) == []


# --- finding overlaps -------------------------------------------------------------


def three_crops(tmp_path):
    folder = write_results(
        tmp_path / "p",
        {
            "left": [crop("l", b=(0.25, 0.5, 0.5, 1.0))],
            "middle": [crop("m", b=(0.5, 0.5, 0.5, 1.0))],
            "right": [crop("r", b=(0.9, 0.5, 0.1, 1.0))],
        },
    )
    return script.load_crops(folder / "results.yaml")[0]


def test_every_overlapping_pair_is_found_and_ordered_by_share(tmp_path):
    found = script.find_overlaps(three_crops(tmp_path))
    assert [(o.a.id, o.b.id) for o in found] == [("l", "m")]


def test_thresholds_filter_small_overlaps(tmp_path):
    crops = three_crops(tmp_path)
    assert script.find_overlaps(crops, min_overlap=0.6) == []
    assert len(script.find_overlaps(crops, min_overlap=0.5)) == 1
    assert script.find_overlaps(crops, min_iou=0.5) == []
    assert len(script.find_overlaps(crops, min_iou=0.3)) == 1


def test_crops_of_different_images_are_never_compared(tmp_path):
    first = write_results(
        tmp_path / "a", {"n": [crop("a1", b=(0.5, 0.5, 1.0, 1.0))]}, image="one.png"
    )
    second = write_results(
        tmp_path / "b", {"n": [crop("b1", b=(0.5, 0.5, 1.0, 1.0))]}, image="one.png"
    )
    crops = (
        script.load_crops(first / "results.yaml")[0]
        + script.load_crops(second / "results.yaml")[0]
    )
    assert script.find_overlaps(crops) == []  # same file name, different images


def test_same_node_only_ignores_overlaps_between_different_nodes(tmp_path):
    crops = three_crops(tmp_path)
    assert script.find_overlaps(crops, same_node_only=True) == []
    folder = write_results(
        tmp_path / "q",
        {"n": [crop("a", b=(0.25, 0.5, 0.5, 1.0)), crop("b", b=(0.5, 0.5, 0.5, 1.0))]},
    )
    same = script.load_crops(folder / "results.yaml")[0]
    assert len(script.find_overlaps(same, same_node_only=True)) == 1


def test_lineage_pairs_can_be_ignored(tmp_path):
    folder = write_results(
        tmp_path / "p",
        {
            "page": [crop("page", b=(0.5, 0.5, 1.0, 1.0))],
            "line": [{**crop("line", b=(0.5, 0.5, 0.5, 0.2)), "parents": "page"}],
        },
    )
    crops = script.load_crops(folder / "results.yaml")[0]
    assert len(script.find_overlaps(crops)) == 1
    assert script.find_overlaps(crops, ignore_lineage=True) == []


# --- the command line -------------------------------------------------------------


def run(args, capsys):
    code = script.main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_exit_status_and_report_when_boxes_overlap(tmp_path, capsys):
    folder = write_results(
        tmp_path / "Page1.livy",
        {
            "boxes": [
                crop("b1", b=(0.25, 0.5, 0.5, 1.0)),
                crop("b2", b=(0.5, 0.5, 0.5, 1.0)),
            ]
        },
    )
    code, out = run([folder], capsys)
    assert code == 1
    assert "Page1.livy" in out
    assert "b1" in out and "b2" in out
    assert "33.3%" in out  # the IoU


def test_exit_status_zero_and_a_clear_message_when_nothing_overlaps(tmp_path, capsys):
    folder = write_results(
        tmp_path / "ok.livy",
        {
            "boxes": [
                crop("b1", b=(0.2, 0.5, 0.2, 0.5)),
                crop("b2", b=(0.7, 0.5, 0.2, 0.5)),
            ]
        },
    )
    code, out = run([folder], capsys)
    assert code == 0
    assert "No overlapping" in out


def test_json_output_is_machine_readable(tmp_path, capsys):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("b1", b=(0.25, 0.5, 0.5, 1.0)),
                crop("b2", b=(0.5, 0.5, 0.5, 1.0)),
            ]
        },
    )
    code, out = run([folder, "--json"], capsys)
    (record,) = json.loads(out)
    assert record["file"].endswith("results.yaml")
    assert {record["a"]["id"], record["b"]["id"]} == {"b1", "b2"}
    assert record["share_of_smaller"] == pytest.approx(0.5)
    assert record["iou"] == pytest.approx(1 / 3)
    assert code == 1


def test_options_reach_the_search(tmp_path, capsys):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("b1", b=(0.25, 0.5, 0.5, 1.0)),
                crop("b2", b=(0.5, 0.5, 0.5, 1.0)),
            ]
        },
    )
    assert run([folder, "--min-overlap", "0.9"], capsys)[0] == 0
    assert run([folder, "--min-iou", "0.9"], capsys)[0] == 0
    assert run([folder, "--same-node-only"], capsys)[0] == 1


def test_no_results_files_is_an_error(tmp_path, capsys):
    code, out = run([tmp_path / "empty"], capsys)
    assert code == 2


def test_skipped_crops_are_mentioned(tmp_path, capsys):
    folder = write_results(
        tmp_path / "p.livy", {"boxes": [crop("ok", b=(0.5, 0.5, 0.2, 0.2))]}
    )
    data = yaml.safe_load((folder / "results.yaml").read_text())
    data["lost"] = {
        "type": "collectra.ImageCrop",
        "id": "lost",
        "data": "gone.png",
        **box(0.5, 0.5, 0.1, 0.1),
    }
    (folder / "results.yaml").write_text(yaml.safe_dump(data))
    code, out = run([folder], capsys)
    assert "could not be loaded" in out


# --- --remove-smaller -----------------------------------------------------------------


def two_overlapping(tmp_path, node_a="boxes", node_b="boxes", **extra):
    """A big crop and a smaller one inside it, in the given nodes."""
    nodes = {node_a: [crop("big", b=(0.5, 0.5, 0.8, 0.8))]}
    nodes.setdefault(node_b, [])
    nodes[node_b] = nodes[node_b] + [crop("small", b=(0.5, 0.5, 0.2, 0.2))]
    nodes.update(extra)
    return write_results(tmp_path / "p.livy", nodes)


def saved(folder, key):
    data = yaml.safe_load((folder / "results.yaml").read_text())
    value = data[key]
    return value if isinstance(value, list) else [value]


def ids_of(folder, key):
    return [item["id"] for item in saved(folder, key)]


def test_the_smaller_box_is_the_one_removed(tmp_path):
    crops = script.load_crops(two_overlapping(tmp_path) / "results.yaml")[0]
    (plan,) = script.plan_removals(script.find_overlaps(crops))
    assert (plan.removed.id, plan.kept.id) == ("small", "big")


def test_equal_boxes_remove_the_lower_confidence_then_the_later_id(tmp_path):
    def pair(first, second):
        folder = write_results(
            tmp_path / f"p{first}{second}",
            {
                "n": [
                    {**crop("a", b=(0.5, 0.5, 0.4, 0.4)), **first},
                    {**crop("b", b=(0.5, 0.5, 0.4, 0.4)), **second},
                ]
            },
        )
        crops = script.load_crops(folder / "results.yaml")[0]
        (plan,) = script.plan_removals(script.find_overlaps(crops))
        return plan.removed.id

    assert pair({"confidence": 0.9}, {"confidence": 0.4}) == "b"
    assert pair({"confidence": 0.2}, {"confidence": 0.8}) == "a"
    assert pair({}, {}) == "b"  # no information: the later id goes


def test_after_planning_no_two_kept_crops_overlap(tmp_path):
    folder = write_results(
        tmp_path / "chain",
        {
            "n": [
                crop("huge", b=(0.5, 0.5, 0.9, 0.9)),
                crop("mid", b=(0.3, 0.3, 0.3, 0.3)),
                crop("tiny", b=(0.25, 0.25, 0.1, 0.1)),
                crop("apart", b=(0.985, 0.985, 0.02, 0.02)),
            ]
        },
    )
    crops = script.load_crops(folder / "results.yaml")[0]
    plans = script.plan_removals(script.find_overlaps(crops))
    gone = {p.removed.id for p in plans}
    kept = [c for c in crops if c.id not in gone]
    assert script.find_overlaps(kept) == []
    assert gone == {"mid", "tiny"}  # the big box and the far-away one stay


def test_removal_edits_the_results_file_and_keeps_everything_else(tmp_path):
    folder = two_overlapping(
        tmp_path,
        note=[{"type": "collectra.Text", "id": "note", "data": "keep me"}],
    )
    code = script.main([str(folder), "--remove-smaller"])
    assert code == 0
    assert ids_of(folder, "boxes") == ["big"]
    assert saved(folder, "note")[0]["data"] == "keep me"
    data = yaml.safe_load((folder / "results.yaml").read_text())
    assert data["collectra_results_metadata"]["workflow"] == "t"
    # one item left is written as a plain mapping, as collectra itself does
    assert isinstance(data["boxes"], dict)


def test_a_node_left_with_none_is_kept_as_empty(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "left": [crop("a", b=(0.5, 0.5, 0.8, 0.8))],
            "right": [crop("b", b=(0.5, 0.5, 0.2, 0.2))],
        },
    )
    script.main([str(folder), "--remove-smaller"])
    data = yaml.safe_load((folder / "results.yaml").read_text())
    assert data["right"] == []  # the key stays: the node ran and has nothing left
    assert data["left"]["id"] == "a"


def test_other_items_in_the_node_keep_their_order_as_a_list(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "n": [
                crop("first", b=(0.1, 0.1, 0.1, 0.1)),
                crop("big", b=(0.6, 0.6, 0.6, 0.6)),
                crop("small", b=(0.6, 0.6, 0.1, 0.1)),
                crop("last", b=(0.95, 0.1, 0.05, 0.05)),
            ]
        },
    )
    script.main([str(folder), "--remove-smaller"])
    assert ids_of(folder, "n") == ["first", "big", "last"]


def test_the_result_loads_and_has_no_overlaps_left(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    script.main([str(folder), "--remove-smaller"])
    capsys.readouterr()
    crops, skipped = script.load_crops(folder / "results.yaml")
    assert [c.id for c in crops] == ["big"] and skipped == 0
    assert script.main([str(folder)]) == 0


def test_same_node_only_removes_only_within_a_node(tmp_path):
    folder = two_overlapping(tmp_path, node_a="outer", node_b="inner")
    code = script.main([str(folder), "--remove-smaller", "--same-node-only"])
    assert ids_of(folder, "outer") == ["big"]
    assert ids_of(folder, "inner") == ["small"]  # an overlap across nodes: left alone
    assert code == 0  # nothing within a node was left to resolve


def test_same_node_only_still_removes_inside_a_node(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("big", b=(0.5, 0.5, 0.8, 0.8)),
                crop("small", b=(0.5, 0.5, 0.2, 0.2)),
            ],
            "other": [crop("elsewhere", b=(0.5, 0.5, 0.1, 0.1))],
        },
    )
    script.main([str(folder), "--remove-smaller", "--same-node-only"])
    assert ids_of(folder, "boxes") == ["big"]
    assert ids_of(folder, "other") == [
        "elsewhere"
    ]  # overlaps big, but a different node


def test_without_the_flag_nothing_is_written(tmp_path):
    folder = two_overlapping(tmp_path)
    before = (folder / "results.yaml").read_text()
    assert script.main([str(folder)]) == 1
    assert (folder / "results.yaml").read_text() == before


def test_dry_run_shows_the_plan_and_writes_nothing(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    before = (folder / "results.yaml").read_text()
    code = script.main([str(folder), "--remove-smaller", "--dry-run"])
    out = capsys.readouterr().out
    assert (folder / "results.yaml").read_text() == before
    assert "Would remove" in out and "small" in out
    assert code == 1


def test_a_removal_report_names_what_was_removed_and_kept(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    script.main([str(folder), "--remove-smaller"])
    out = capsys.readouterr().out
    assert "Removed 1 crop" in out
    assert "small" in out and "big" in out


def test_crops_other_items_depend_on_are_not_removed(tmp_path, capsys):
    folder = two_overlapping(
        tmp_path,
        words=[
            {
                "type": "collectra.Text",
                "id": "words",
                "data": "read from it",
                "parents": "small",
            }
        ],
    )
    code = script.main([str(folder), "--remove-smaller"])
    out = capsys.readouterr().out
    assert ids_of(folder, "boxes") == ["big", "small"]
    assert "words" in out and "depend" in out
    assert code == 1  # an overlap is left


def test_dependents_listed_as_a_list_of_parents_also_block(tmp_path):
    folder = two_overlapping(
        tmp_path,
        words=[
            {
                "type": "collectra.Text",
                "id": "words",
                "data": "x",
                "parents": ["other", "small"],
            }
        ],
    )
    script.main([str(folder), "--remove-smaller"])
    assert ids_of(folder, "boxes") == ["big", "small"]


def test_dependents_that_are_removed_too_do_not_block(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "n": [
                crop("big", b=(0.5, 0.5, 0.9, 0.9)),
                crop("mid", b=(0.4, 0.4, 0.3, 0.3)),
                {**crop("tiny", b=(0.4, 0.4, 0.1, 0.1)), "parents": "mid"},
            ]
        },
    )
    script.main([str(folder), "--remove-smaller"])
    assert ids_of(folder, "n") == ["big"]


def test_each_results_file_is_handled_on_its_own(tmp_path, capsys):
    first = two_overlapping(tmp_path / "one")
    second = write_results(
        tmp_path / "two" / "q.livy", {"n": [crop("solo", b=(0.5, 0.5, 0.2, 0.2))]}
    )
    code = script.main([str(tmp_path), "--remove-smaller"])
    assert ids_of(first, "boxes") == ["big"]
    assert ids_of(second, "n") == ["solo"]
    assert code == 0


def test_json_reports_what_was_removed(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    script.main([str(folder), "--remove-smaller", "--json"])
    (record,) = json.loads(capsys.readouterr().out)
    assert record["removed"] == {"node": "boxes", "id": "small"}
    assert record["kept"] == {"node": "boxes", "id": "big"}
    assert record["applied"] is True
    assert ids_of(folder, "boxes") == ["big"]


def test_json_without_removal_has_no_removed_crop(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    script.main([str(folder), "--json"])
    (record,) = json.loads(capsys.readouterr().out)
    assert record["removed"] is None and record["applied"] is False


def test_dry_run_requires_remove_smaller(tmp_path):
    with pytest.raises(SystemExit):
        script.main([str(two_overlapping(tmp_path)), "--dry-run"])


# --- the tables are kept short: ids and IoU only ---------------------------------------


def test_the_overlap_table_shows_only_ids_and_iou(tmp_path, capsys):
    folder = write_results(
        tmp_path / "Page1.livy",
        {
            "first_node": [crop("b1", b=(0.25, 0.5, 0.5, 1.0))],
            "second_node": [crop("b2", b=(0.5, 0.5, 0.5, 1.0))],
        },
    )
    code, out = run([folder], capsys)
    header = [line for line in out.splitlines() if "Crop A" in line][0]
    assert all(name in header for name in ("Results", "Crop A", "Crop B", "IoU"))
    assert "Of smaller" not in out and "Overlap (px" not in out
    assert "first_node" not in out and "second_node" not in out
    assert "b1" in out and "b2" in out
    assert "(" not in out.split("IoU")[1].splitlines()[2]  # no pixel box in the rows
    assert "50.0%" not in out  # the share-of-smaller figure is gone


def test_the_removal_tables_show_only_ids_and_iou(tmp_path, capsys):
    folder = two_overlapping(tmp_path, node_a="outer_node", node_b="inner_node")
    code, out = run([folder, "--remove-smaller", "--dry-run"], capsys)
    assert "outer_node" not in out and "inner_node" not in out
    assert "Of smaller" not in out
    assert "big" in out and "small" in out and "IoU" in out
    assert "6.2%" in out  # 0.04 / 0.64: the small box inside the big one


def test_the_blocked_table_shows_ids_not_node_names(tmp_path, capsys):
    folder = two_overlapping(
        tmp_path,
        node_a="outer_node",
        node_b="inner_node",
        words=[
            {"type": "collectra.Text", "id": "words", "data": "x", "parents": "small"}
        ],
    )
    code, out = run([folder, "--remove-smaller"], capsys)
    assert "outer_node" not in out and "inner_node" not in out
    assert "small" in out and "big" in out and "words" in out


def test_json_keeps_the_full_detail(tmp_path, capsys):
    folder = two_overlapping(tmp_path)
    run([folder, "--json"], capsys)
    code, out = run([folder, "--json"], capsys)
    (record,) = json.loads(out)
    assert record["a"]["node"] == "boxes" and "share_of_smaller" in record
    assert "intersection_pixels" in record


# --- --min-iou with --remove-smaller ------------------------------------------------------


def half_overlapping(tmp_path):
    """Two equal-height boxes sharing a quarter of the width: IoU is 1/3."""
    return write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("left", b=(0.25, 0.5, 0.5, 1.0)),
                crop("right", b=(0.5, 0.5, 0.5, 1.0)),
            ]
        },
    )


def test_min_iou_above_the_overlap_removes_nothing(tmp_path, capsys):
    folder = half_overlapping(tmp_path)
    before = (folder / "results.yaml").read_text()
    code = script.main([str(folder), "--remove-smaller", "--min-iou", "0.5"])
    assert (folder / "results.yaml").read_text() == before
    assert code == 0  # an overlap below the threshold is not one to resolve


def test_min_iou_at_or_below_the_overlap_removes_the_smaller(tmp_path):
    folder = half_overlapping(tmp_path)
    assert script.main([str(folder), "--remove-smaller", "--min-iou", "0.3"]) == 0
    assert len(saved(folder, "boxes")) == 1


def test_min_iou_only_resolves_the_pairs_that_reach_it(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("a", b=(0.20, 0.5, 0.30, 1.0)),  # a and b: IoU 0.5
                crop("b", b=(0.30, 0.5, 0.30, 1.0)),
                crop("c", b=(0.60, 0.5, 0.20, 1.0)),  # c and d: IoU 1/7
                crop("d", b=(0.75, 0.5, 0.20, 1.0)),
            ]
        },
    )
    script.main([str(folder), "--remove-smaller", "--min-iou", "0.4"])
    assert ids_of(folder, "boxes") == ["a", "c", "d"]  # b went; c and d stay


def test_min_iou_combines_with_same_node_only(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "left": [crop("l", b=(0.25, 0.5, 0.5, 1.0))],
            "right": [crop("r", b=(0.5, 0.5, 0.5, 1.0))],
            "twins": [
                crop("t1", b=(0.5, 0.5, 0.5, 0.5)),
                crop("t2", b=(0.5, 0.5, 0.5, 0.5)),
            ],
        },
    )
    script.main(
        [str(folder), "--remove-smaller", "--same-node-only", "--min-iou", "0.9"]
    )
    assert ids_of(folder, "left") == ["l"] and ids_of(folder, "right") == ["r"]
    assert ids_of(folder, "twins") == ["t1"]  # identical boxes: IoU 1.0, same node


def test_a_box_inside_another_has_a_low_iou_so_min_iou_leaves_it(tmp_path):
    folder = two_overlapping(tmp_path)  # a 0.2 box inside a 0.8 box: IoU 0.0625
    script.main([str(folder), "--remove-smaller", "--min-iou", "0.5"])
    assert ids_of(folder, "boxes") == ["big", "small"]
    script.main([str(folder), "--remove-smaller", "--min-overlap", "0.9"])
    assert ids_of(folder, "boxes") == ["big"]  # min-overlap catches containment


# --- --repoint-dependents --------------------------------------------------------------------


def near_duplicates(tmp_path, parents="small", **extra):
    """A big crop and a slightly smaller one (IoU 0.95) with something read from the smaller."""
    return write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("big", b=(0.5, 0.5, 0.8, 0.8)),
                crop("small", b=(0.5, 0.5, 0.78, 0.78)),
            ],
            "words": [
                {
                    "type": "collectra.Text",
                    "id": "words",
                    "data": "read from small",
                    "parents": parents,
                }
            ],
            **extra,
        },
    )


def parents_of(folder, key="words"):
    return saved(folder, key)[0]["parents"]


def test_dependents_move_to_the_crop_that_is_kept(tmp_path, capsys):
    folder = near_duplicates(tmp_path)
    code = script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    assert code == 0
    assert ids_of(folder, "boxes") == ["big"]
    assert parents_of(folder) == "big"
    assert (
        saved(folder, "words")[0]["data"] == "read from small"
    )  # nothing else changed


def test_a_list_of_parents_is_repointed_in_place_without_duplicates(tmp_path):
    folder = near_duplicates(tmp_path, parents=["other", "small", "big"])
    script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    assert parents_of(folder) == ["other", "big"]


def test_without_the_flag_dependents_still_block(tmp_path):
    folder = near_duplicates(tmp_path)
    before = (folder / "results.yaml").read_text()
    assert script.main([str(folder), "--remove-smaller"]) == 1
    assert (folder / "results.yaml").read_text() == before


def test_a_low_iou_pair_is_not_repointed_by_default(tmp_path, capsys):
    folder = two_overlapping(
        tmp_path,
        words=[
            {"type": "collectra.Text", "id": "words", "data": "x", "parents": "small"}
        ],
    )  # a small box inside a big one: IoU 0.0625
    before = (folder / "results.yaml").read_text()
    code = script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    out = capsys.readouterr().out
    assert (folder / "results.yaml").read_text() == before
    assert code == 1
    assert "re-point" in out.lower() and "90%" in out


def test_the_minimum_iou_for_repointing_can_be_lowered(tmp_path):
    folder = two_overlapping(
        tmp_path,
        words=[
            {"type": "collectra.Text", "id": "words", "data": "x", "parents": "small"}
        ],
    )
    script.main(
        [
            str(folder),
            "--remove-smaller",
            "--repoint-dependents",
            "--repoint-min-iou",
            "0.05",
        ]
    )
    assert ids_of(folder, "boxes") == ["big"]
    assert parents_of(folder) == "big"


def test_dependents_follow_a_chain_of_removals_to_the_final_survivor(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("M", b=(0.5, 0.5, 0.90, 0.90)),
                crop("K", b=(0.5, 0.5, 0.88, 0.88)),
                crop("A", b=(0.5, 0.5, 0.86, 0.86)),
            ],
            "from_a": [
                {"type": "collectra.Text", "id": "from_a", "data": "a", "parents": "A"}
            ],
            "from_k": [
                {
                    "type": "collectra.Text",
                    "id": "from_k",
                    "data": "k",
                    "parents": ["K"],
                }
            ],
        },
    )
    script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    assert ids_of(folder, "boxes") == ["M"]
    assert parents_of(folder, "from_a") == "M"
    assert parents_of(folder, "from_k") == ["M"]


def test_crops_without_dependents_are_removed_alongside_repointed_ones(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "boxes": [
                crop("big", b=(0.5, 0.5, 0.9, 0.9)),
                crop("mid", b=(0.5, 0.5, 0.88, 0.88)),  # near-duplicate of big
                crop("tiny", b=(0.5, 0.5, 0.2, 0.2)),  # inside both: low IoU
            ],
            "from_mid": [
                {
                    "type": "collectra.Text",
                    "id": "from_mid",
                    "data": "m",
                    "parents": "mid",
                }
            ],
        },
    )
    script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    assert ids_of(folder, "boxes") == ["big"]  # tiny has no dependents, so it goes too
    assert parents_of(folder, "from_mid") == "big"


def test_dry_run_lists_what_would_be_repointed_and_writes_nothing(tmp_path, capsys):
    folder = near_duplicates(tmp_path)
    before = (folder / "results.yaml").read_text()
    code = script.main(
        [str(folder), "--remove-smaller", "--repoint-dependents", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert (folder / "results.yaml").read_text() == before
    assert "words" in out and "Would remove" in out
    assert code == 1


def test_json_names_the_repointed_items(tmp_path, capsys):
    folder = near_duplicates(tmp_path)
    script.main([str(folder), "--remove-smaller", "--repoint-dependents", "--json"])
    (record,) = json.loads(capsys.readouterr().out)
    assert record["repointed"] == ["words"]
    assert record["removed"]["id"] == "small" and record["applied"] is True


def test_repointing_respects_same_node_only(tmp_path):
    folder = write_results(
        tmp_path / "p.livy",
        {
            "outer": [crop("big", b=(0.5, 0.5, 0.8, 0.8))],
            "inner": [crop("small", b=(0.5, 0.5, 0.78, 0.78))],
            "words": [
                {
                    "type": "collectra.Text",
                    "id": "words",
                    "data": "x",
                    "parents": "small",
                }
            ],
        },
    )
    script.main(
        [str(folder), "--remove-smaller", "--repoint-dependents", "--same-node-only"]
    )
    assert ids_of(folder, "inner") == ["small"]  # a different node: nothing to resolve
    assert parents_of(folder) == "small"


def test_the_result_still_loads_with_its_lineage_intact(tmp_path):
    from collectra.types.base import ArtefactNode
    from collectra.types.texts import Text

    folder = near_duplicates(tmp_path)
    script.main([str(folder), "--remove-smaller", "--repoint-dependents"])
    crops, skipped = script.load_crops(folder / "results.yaml")
    words = ArtefactNode("words", types={Text})
    words.process("words", folder)
    (word,) = words.items.values()
    assert [c.id for c in crops] == ["big"] and skipped == 0
    assert word.parents == ["big"]


def test_the_repoint_options_need_remove_smaller(tmp_path):
    folder = near_duplicates(tmp_path)
    with pytest.raises(SystemExit):
        script.main([str(folder), "--repoint-dependents"])
    with pytest.raises(SystemExit):
        script.main([str(folder), "--repoint-min-iou", "0.5"])


# --- the results header ----------------------------------------------------------


def test_the_new_results_header_is_ignored(tmp_path):
    folder = write_results(
        tmp_path / "p",
        {
            "n": [
                crop("c1", b=(0.5, 0.5, 0.5, 0.5)),
                crop("c2", b=(0.5, 0.5, 0.5, 0.5)),
            ]
        },
    )
    path = folder / "results.yaml"
    data = yaml.safe_load(path.read_text())
    data.pop("collectra_results_metadata")
    path.write_text(
        yaml.safe_dump(
            {"results": {"type": "collectra.Results", "partition": "p"}, **data},
            sort_keys=False,
        )
    )
    crops, failed = script.load_crops(path)
    assert [c.id for c in crops] == ["c1", "c2"] and failed == 0
    assert script.dependents_map(yaml.safe_load(path.read_text())) == {}
