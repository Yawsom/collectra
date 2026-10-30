"""New artefacts are named after their node: ``node`` when alone, ``node-k`` otherwise."""

from pathlib import Path

import pytest
import yaml

from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.test_pipeline_stages import build, files  # noqa: F401

THIS = "tests.unit.test_artefact_ids"


class Split(Task):
    """Produces several outputs per image."""

    def run(self, image: Image) -> list[Text]:
        stem = Path(image.get_path()).stem
        return [
            Text(name=self.get_output_name(), data=f"{stem} part {n}")
            for n in (1, 2, 3)
        ]


class Echo(Task):
    """One output per input, so each output has exactly one parent."""

    def run(self, text: Text) -> Text:
        return Text(name=self.get_output_name(), data=text().upper())


class Fixed(Task):
    """Chooses its own ids."""

    def run(self, image: Image) -> list[Text]:
        return [Text(name=self.get_output_name(), id="chosen", data="x")]


class Repeat(Task):
    """Emits two identical outputs."""

    def run(self, image: Image) -> list[Text]:
        return [Text(name=self.get_output_name(), data="same") for _ in range(2)]


def pipeline(tmp_path, **tasks):
    from collectra import Collectra

    return Collectra("Ids", "ids", "0.1.0", path=str(tmp_path), **tasks)


def results(tmp_path, name="a"):
    return yaml.safe_load(
        (tmp_path / "input" / f"{name}.ids" / "results.yaml").read_text()
    )


def items(data, key):
    value = data[key]
    return value if isinstance(value, list) else [value]


def multi(tmp_path):
    return pipeline(
        tmp_path,
        split={
            "type": f"{THIS}.Split",
            "input": ["page"],
            "output": ["parts"],
        },
        echo={
            "type": f"{THIS}.Echo",
            "input": ["parts"],
            "output": ["echoed"],
        },
    )


def test_a_single_artefact_is_named_after_its_node(tmp_path, files):
    build(tmp_path).run(files=files[:1])
    data = yaml.safe_load((tmp_path / "input" / "a.stg" / "results.yaml").read_text())
    assert items(data, "described")[0]["id"] == "described"
    shouted = items(data, "shouted")[0]
    assert shouted["id"] == "shouted"
    assert shouted["parents"] == "described"


def test_several_artefacts_are_numbered_from_one(tmp_path, files):
    multi(tmp_path).run(files=files[:1])
    data = results(tmp_path)
    assert [i["id"] for i in items(data, "parts")] == ["parts-1", "parts-2", "parts-3"]
    assert [i["data"] for i in items(data, "parts")] == [
        f"a part {n}" for n in (1, 2, 3)
    ]


def test_children_reference_the_final_ids_of_their_parents(tmp_path, files):
    multi(tmp_path).run(files=files[:1])
    echoed = items(results(tmp_path), "echoed")
    assert [i["id"] for i in echoed] == ["echoed-1", "echoed-2", "echoed-3"]
    assert [i["parents"] for i in echoed] == ["parts-1", "parts-2", "parts-3"]


def test_ids_a_task_chose_itself_are_kept(tmp_path, files):
    pipeline(
        tmp_path,
        fixed={
            "type": f"{THIS}.Fixed",
            "input": ["page"],
            "output": ["fixed_out"],
        },
    ).run(files=files[:1])
    assert items(results(tmp_path), "fixed_out")[0]["id"] == "chosen"


def test_identical_outputs_still_collapse_into_one_named_after_the_node(
    tmp_path, files
):
    pipeline(
        tmp_path,
        repeat={
            "type": f"{THIS}.Repeat",
            "input": ["page"],
            "output": ["same_out"],
        },
    ).run(files=files[:1])
    found = items(results(tmp_path), "same_out")
    assert [i["id"] for i in found] == ["same_out"]


def test_rerunning_with_force_reproduces_the_same_ids(tmp_path, files):
    first = multi(tmp_path)
    first.run(files=files[:1])
    folder = str(tmp_path / "input" / "a.ids")
    multi(tmp_path).run(files=[folder], force=True)
    data = results(tmp_path)
    assert [i["id"] for i in items(data, "parts")] == ["parts-1", "parts-2", "parts-3"]
    assert [i["parents"] for i in items(data, "echoed")] == [
        "parts-1",
        "parts-2",
        "parts-3",
    ]


def test_new_items_beside_existing_ones_take_the_next_free_number(tmp_path):
    pipe = pipeline(
        tmp_path,
        echo={
            "type": f"{THIS}.Echo",
            "input": ["parts"],
            "output": ["echoed"],
        },
    )
    pipe.connect()
    node = pipe.node_manager.resolve_node("echoed")
    node.add_item(Text(name="echoed", id="echoed", data="old"))
    pipe._execute_artefact_node(node, (Text(name="echoed", data="new"),), {}, False)
    assert sorted(node.items) == ["echoed", "echoed-1"]
    assert node.items["echoed-1"].data == "new"


def test_references_in_other_results_of_the_same_batch_follow_the_rename(tmp_path):
    pipe = pipeline(
        tmp_path,
        echo={
            "type": f"{THIS}.Echo",
            "input": ["parts"],
            "output": ["echoed"],
        },
    )
    pipe.connect()
    node = pipe.node_manager.resolve_node("echoed")
    produced = Text(name="echoed", data="a")
    sibling = Text(name="other", data="b", parents=[produced.id])
    pipe._execute_artefact_node(node, (produced, sibling), {}, False)
    assert produced.id == "echoed"
    assert sibling.parents == ["echoed"]


def test_automatic_ids_are_not_written_to_results(tmp_path, files):
    build(tmp_path).run(files=files[:1])
    text = (tmp_path / "input" / "a.stg" / "results.yaml").read_text()
    assert "_auto_id" not in text


def test_an_identical_rerun_keeps_the_existing_items_id(tmp_path):
    pipe = pipeline(
        tmp_path,
        echo={"type": f"{THIS}.Echo", "input": ["parts"], "output": ["echoed"]},
    )
    pipe.connect()
    node = pipe.node_manager.resolve_node("echoed")
    node.add_item(Text(name="echoed", id="echoed-abc123", data="same"))
    pipe._execute_artefact_node(node, (Text(name="echoed", data="same"),), {}, False)
    assert list(node.items) == ["echoed-abc123"]


def test_equal_looking_items_are_each_renamed_not_confused_with_existing_ones(tmp_path):
    pipe = pipeline(
        tmp_path,
        echo={"type": f"{THIS}.Echo", "input": ["parts"], "output": ["echoed"]},
    )
    pipe.connect()
    node = pipe.node_manager.resolve_node("echoed")
    kept = Text(name="echoed", id="echoed-kept", data="x", parents=["p"])
    node.add_item(kept)
    twin = Text(name="echoed", data="x", parents=["q"])
    pipe._execute_artefact_node(node, (twin,), {}, False)
    assert sorted(node.items) == ["echoed-1", "echoed-kept"]
