"""Text-producing tasks can save each output to ``{node}/{id}.{extension}``."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.tasks.text_output import TextOutputTask
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.test_pipeline_stages import Describe, files  # noqa: F401

THIS = "tests.unit.test_text_output"


class Writer(TextOutputTask):
    """Writes three texts per image."""

    def run(self, image: Image) -> list[Text]:
        stem = Path(image.get_path()).stem
        return [self.text_output(f"{stem} part {n}") for n in (1, 2, 3)]


class One(TextOutputTask):
    def run(self, image: Image) -> Text:
        return self.text_output(f"only {Path(image.get_path()).stem}")


class Shout(Task):
    """An ordinary task reading the (possibly file-backed) text."""

    def run(self, text: Text) -> Text:
        return Text(name=self.get_output_name(), data=text().upper())


def pipeline(tmp_path, **tasks):
    return Pipeline("Out", "out", "0.1.0", path=str(tmp_path), **tasks)


def folder(tmp_path, name="a"):
    return tmp_path / "input" / f"{name}.out"


def record(tmp_path, key, name="a"):
    data = yaml.safe_load((folder(tmp_path, name) / "results.yaml").read_text())[key]
    return data if isinstance(data, list) else [data]


# --- the extension option --------------------------------------------------


@pytest.mark.parametrize(
    "given, kept",
    [
        ("md", "md"),
        (".md", "md"),
        ("XML", "xml"),
        (" txt ", "txt"),
        (None, None),
        ("", None),
    ],
)
def test_output_extension_is_normalised(given, kept):
    assert One("t", output_extension=given).output_extension == kept


@pytest.mark.parametrize("bad", ["m/d", "..", "a b", "x.y", "é"])
def test_output_extension_must_be_a_plain_suffix(bad):
    with pytest.raises(ValidationError):
        One("t", output_extension=bad)


def test_output_extension_is_off_by_default_and_declared_as_a_field():
    assert TextOutputTask.model_fields["output_extension"].default is None
    assert One("t").output_extension is None


def test_text_output_marks_the_text_without_writing_anything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = One("t", output_extension="md").text_output("hello")
    assert isinstance(text, Text)
    assert text() == "hello"
    assert text.name == "t_output"
    assert list(tmp_path.iterdir()) == []


# --- the pipeline writes the file once the id is final ---------------------


def one(tmp_path, **extra):
    return pipeline(
        tmp_path,
        one={"type": f"{THIS}.One", "input": ["page"], "output": ["answer"], **extra},
    )


def many(tmp_path, **extra):
    return pipeline(
        tmp_path,
        writer={
            "type": f"{THIS}.Writer",
            "input": ["page"],
            "output": ["parts"],
            **extra,
        },
    )


def test_a_single_output_is_saved_under_the_node_name(tmp_path, files):
    one(tmp_path, output_extension="md").run(files=files[:1])
    assert (folder(tmp_path) / "answer" / "answer.md").read_text() == "only a"
    (item,) = record(tmp_path, "answer")
    assert item["id"] == "answer"
    assert item["path"] == "answer/answer.md"
    assert "data" not in item


def test_several_outputs_are_numbered_files(tmp_path, files):
    many(tmp_path, output_extension="txt").run(files=files[:1])
    names = sorted(p.name for p in (folder(tmp_path) / "parts").iterdir())
    assert names == ["parts-1.txt", "parts-2.txt", "parts-3.txt"]
    assert (folder(tmp_path) / "parts" / "parts-2.txt").read_text() == "a part 2"
    assert [i["path"] for i in record(tmp_path, "parts")] == [
        "parts/parts-1.txt", "parts/parts-2.txt", "parts/parts-3.txt",
    ]  # fmt: skip


def test_any_extension_works(tmp_path, files):
    one(tmp_path, output_extension="xml").run(files=files[:1])
    assert (folder(tmp_path) / "answer" / "answer.xml").read_text() == "only a"


def test_without_an_extension_the_text_stays_inline(tmp_path, files):
    one(tmp_path).run(files=files[:1])
    (item,) = record(tmp_path, "answer")
    assert item["data"] == "only a"
    assert "path" not in item
    assert not (folder(tmp_path) / "answer").exists()


def test_later_tasks_still_read_the_text(tmp_path, files):
    pipe = pipeline(
        tmp_path,
        writer={
            "type": f"{THIS}.Writer",
            "input": ["page"],
            "output": ["parts"],
            "output_extension": "md",
        },
        shout={"type": f"{THIS}.Shout", "input": ["parts"], "output": ["loud"]},
    )
    pipe.run(files=files[:1])
    assert [i["data"] for i in record(tmp_path, "loud")] == [
        "A PART 1", "A PART 2", "A PART 3",
    ]  # fmt: skip


def test_a_task_resumed_from_saved_results_reads_the_files(tmp_path, files):
    pipe = pipeline(
        tmp_path,
        writer={
            "type": f"{THIS}.Writer",
            "input": ["page"],
            "output": ["parts"],
            "output_extension": "md",
        },
        shout={"type": f"{THIS}.Shout", "input": ["parts"], "output": ["loud"]},
    )
    pipe.run(files=files[:1])
    results = folder(tmp_path) / "results.yaml"
    data = yaml.safe_load(results.read_text())
    del data["loud"]
    results.write_text(yaml.safe_dump(data))
    pipe.run("shout", single=True, files=[str(folder(tmp_path))])
    assert [i["data"] for i in record(tmp_path, "loud")][0] == "A PART 1"


def test_rerunning_with_force_rewrites_the_files_under_the_same_names(tmp_path, files):
    many(tmp_path, output_extension="md").run(files=files[:1])
    (folder(tmp_path) / "parts" / "parts-1.md").write_text("edited")
    many(tmp_path, output_extension="md").run(files=[str(folder(tmp_path))], force=True)
    assert (folder(tmp_path) / "parts" / "parts-1.md").read_text() == "a part 1"
    assert [i["id"] for i in record(tmp_path, "parts")] == [
        "parts-1",
        "parts-2",
        "parts-3",
    ]


def test_skipped_tasks_leave_their_files_alone(tmp_path, files):
    many(tmp_path, output_extension="md").run(files=files[:1])
    (folder(tmp_path) / "parts" / "parts-1.md").write_text("edited")
    many(tmp_path, output_extension="md").run(files=[str(folder(tmp_path))])
    assert (folder(tmp_path) / "parts" / "parts-1.md").read_text() == "edited"


def test_ids_are_made_safe_for_file_names(tmp_path):
    text = Text(name="n", id="../evil/name", data="x")
    text._file_extension = "md"
    text.finalise(tmp_path, "node")
    written = list((tmp_path / "node").iterdir())
    assert [p.name for p in written] == ["..__evil_name.md"] or all(
        p.parent == tmp_path / "node" for p in written
    )
    assert not (tmp_path / "evil").exists()


def test_files_are_written_once(tmp_path):
    text = Text(name="n", id="n", data="x")
    text._file_extension = "md"
    text.finalise(tmp_path, "node")
    (tmp_path / "node" / "n.md").write_text("changed on disk")
    text.finalise(tmp_path, "node")
    assert (tmp_path / "node" / "n.md").read_text() == "changed on disk"


# --- the tasks that produce text -------------------------------------------


def test_the_text_tasks_share_the_parent_class():
    from collectra.tasks.canonicalisers import LLMCanonicaliser
    from collectra.tasks.irn_resolvers import IRNResolver
    from collectra.tasks.llms import LLM
    from collectra.tasks.ocr import OCRSuraya
    from collectra.tasks.ocr_huggingface import OCRHuggingFace
    from collectra.tasks.text import ConcatenateText

    for cls in (
        LLM,
        LLMCanonicaliser,
        IRNResolver,
        OCRSuraya,
        OCRHuggingFace,
        ConcatenateText,
    ):
        assert issubclass(cls, TextOutputTask), cls


def test_concatenated_text_can_be_saved(tmp_path):
    from collectra.tasks.text import ConcatenateText

    joined = ConcatenateText("join", separator="+", output_extension="md").run(
        Text(name="a", data="1"), Text(name="b", data="2")
    )
    assert joined() == "1+2"
    joined.id = "joined"
    joined.finalise(tmp_path, "joined")
    assert (tmp_path / "joined" / "joined.md").read_text() == "1+2"


def test_an_llm_saves_its_response_to_a_file(tmp_path, files):
    from collectra.tasks.llms import LLM

    class Chain:
        def invoke(self, messages):
            return AIMessage(content="# Heading\n\nBody")

    with patch.object(LLM, "chain", property(lambda self: Chain())):
        pipe = pipeline(
            tmp_path,
            first={
                "type": f"{THIS}.DescribeText",
                "input": ["page"],
                "output": ["described"],
            },
            llm={
                "type": "collectra.LLM",
                "model": "m",
                "template": "{described}",
                "input": ["described"],
                "output": ["answer"],
                "output_extension": "md",
            },
        )
        pipe.run(files=files[:1])
    assert (
        folder(tmp_path) / "answer" / "answer.md"
    ).read_text() == "# Heading\n\nBody"
    (item,) = record(tmp_path, "answer")
    assert item["path"] == "answer/answer.md"


class DescribeText(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="described")


# --- --force removes the node's own files, exactly as recorded -------------


SEEN: list = []


class Spy(TextOutputTask):
    """Records what the node's folder holds at the moment the task runs."""

    keep: int = 3

    def run(self, image: Image) -> list[Text]:
        node_folder = self.context.file_path / "parts"
        SEEN.append(
            sorted(p.name for p in node_folder.glob("*"))
            if node_folder.exists()
            else []
        )
        return [self.text_output(f"new {n}") for n in range(1, self.keep + 1)]


def spy(tmp_path, **extra):
    return pipeline(
        tmp_path,
        spy={"type": f"{THIS}.Spy", "input": ["page"], "output": ["parts"], **extra},
    )


@pytest.fixture
def done(tmp_path, files):
    """One result folder whose ``parts`` node has three files, beside a stray one."""
    SEEN.clear()
    spy(tmp_path, output_extension="md").run(files=files[:1])
    (folder(tmp_path) / "parts" / "stray.md").write_text("not recorded anywhere")
    SEEN.clear()
    return str(folder(tmp_path))


def test_force_deletes_the_files_the_artefacts_record_before_the_task_runs(
    tmp_path, done
):
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    assert SEEN == [["stray.md"]]  # the three recorded files were gone at that point


def test_files_nobody_recorded_are_never_deleted(tmp_path, done):
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    assert (
        folder(tmp_path) / "parts" / "stray.md"
    ).read_text() == "not recorded anywhere"


def test_force_replaces_the_recorded_files_with_the_new_ones(tmp_path, done):
    spy(tmp_path, output_extension="md", keep=1).run(files=[done], force=True)
    names = sorted(p.name for p in (folder(tmp_path) / "parts").iterdir())
    assert names == ["parts.md", "stray.md"]
    assert (folder(tmp_path) / "parts" / "parts.md").read_text() == "new 1"
    assert [i["path"] for i in record(tmp_path, "parts")] == ["parts/parts.md"]


def test_without_force_nothing_is_deleted(tmp_path, done):
    spy(tmp_path, output_extension="md").run(files=[done])
    assert SEEN == []  # skipped
    assert (folder(tmp_path) / "parts" / "parts-1.md").exists()


def test_other_nodes_and_subfolders_are_left_alone(tmp_path, done):
    other = folder(tmp_path) / "other"
    other.mkdir()
    (other / "keep.md").write_text("mine")
    nested = folder(tmp_path) / "parts" / "nested"
    nested.mkdir()
    (nested / "keep.md").write_text("mine too")
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    assert (other / "keep.md").read_text() == "mine"
    assert (nested / "keep.md").read_text() == "mine too"


def test_a_failed_rerun_puts_the_old_files_back(tmp_path, done, monkeypatch):
    before = {p.name: p.read_text() for p in (folder(tmp_path) / "parts").iterdir()}

    def fail(self, image: Image) -> list[Text]:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(Spy, "run", fail)
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    after = {p.name: p.read_text() for p in (folder(tmp_path) / "parts").iterdir()}
    assert after == before
    assert [i["path"] for i in record(tmp_path, "parts")] == [
        "parts/parts-1.md", "parts/parts-2.md", "parts/parts-3.md",
    ]  # fmt: skip


def test_nothing_is_left_behind_in_temporary_folders(tmp_path, done):
    import tempfile

    before = set(Path(tempfile.gettempdir()).glob("collectra-previous-*"))
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    assert set(Path(tempfile.gettempdir()).glob("collectra-previous-*")) == before


def test_each_result_folder_is_handled_separately(tmp_path, files):
    spy(tmp_path, output_extension="md").run(files=files[:2])
    SEEN.clear()
    spy(tmp_path, output_extension="md").run(
        files=[str(folder(tmp_path, "a")), str(folder(tmp_path, "b"))], force=True
    )
    assert SEEN == [[], []]


# --- results written by hand, or by older versions, are handled as recorded


def handmade(tmp_path, files, entries: dict, extra_files: dict):
    """A results folder crafted by hand: ``entries`` go into results.yaml."""
    spy(tmp_path).run(files=files[:1])  # creates the folder and its page
    results = folder(tmp_path) / "results.yaml"
    data = yaml.safe_load(results.read_text())
    data.pop("parts", None)
    data.update(entries)
    results.write_text(yaml.safe_dump(data))
    for name, text in extra_files.items():
        target = folder(tmp_path) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    SEEN.clear()
    return str(folder(tmp_path))


def text_record(id, **fields):
    return {"type": "collectra.Text", "id": id, "orientation": "north", **fields}


def test_a_file_in_any_layout_is_deleted_by_its_recorded_path(tmp_path, files):
    done = handmade(
        tmp_path,
        files,
        {"parts": text_record("parts", path="old_tei.xml")},
        {"old_tei.xml": "<old/>", "markdown.md": "someone else's"},
    )
    spy(tmp_path, output_extension="md").run(files=[done], force=True)
    assert not (folder(tmp_path) / "old_tei.xml").exists()
    assert (folder(tmp_path) / "markdown.md").read_text() == "someone else's"


def test_items_without_files_are_simply_dropped(tmp_path, files):
    done = handmade(
        tmp_path,
        files,
        {
            "parts": [
                text_record("p1", data="inline one"),
                text_record("p2", data="inline two"),
            ]
        },
        {"keep.txt": "untouched"},
    )
    spy(tmp_path, keep=1).run(files=[done], force=True)
    assert [i["data"] for i in record(tmp_path, "parts")] == ["new 1"]
    assert (folder(tmp_path) / "keep.txt").exists()


def test_a_mixed_node_deletes_only_the_files_it_names(tmp_path, files):
    done = handmade(
        tmp_path,
        files,
        {
            "parts": [
                text_record("a1", data="inline"),
                text_record("a2", path="notes/two.txt"),
                text_record("a3", path="three.md"),
            ]
        },
        {"notes/two.txt": "2", "notes/sibling.txt": "sibling", "three.md": "3"},
    )
    spy(tmp_path, keep=1).run(files=[done], force=True)
    assert not (folder(tmp_path) / "notes" / "two.txt").exists()
    assert not (folder(tmp_path) / "three.md").exists()
    assert (folder(tmp_path) / "notes" / "sibling.txt").read_text() == "sibling"


def test_a_recorded_file_that_is_already_gone_is_not_an_error(tmp_path, files):
    done = handmade(
        tmp_path, files, {"parts": text_record("parts", data="x", path=None)}, {}
    )
    data = yaml.safe_load((folder(tmp_path) / "results.yaml").read_text())
    data["parts"] = text_record("parts", path="missing.md")
    (folder(tmp_path) / "results.yaml").write_text(yaml.safe_dump(data))
    (folder(tmp_path) / "missing.md").write_text("here for loading")
    # Load once, remove the file, then force: the record outlives the file.
    pipe = spy(tmp_path, keep=1)
    original = pipe._take_outputs

    def vanish(node, folder_):
        (folder(tmp_path) / "missing.md").unlink()
        return original(node, folder_)

    pipe._take_outputs = vanish
    pipe.run(files=[done], force=True)
    assert [i["data"] for i in record(tmp_path, "parts")] == ["new 1"]


def test_files_outside_the_results_folder_are_not_deleted(tmp_path, files, caplog):
    outside = tmp_path / "shared" / "tei.xml"
    outside.parent.mkdir()
    outside.write_text("shared by several people")
    done = handmade(
        tmp_path, files, {"parts": text_record("parts", path=str(outside))}, {}
    )
    with caplog.at_level("WARNING", logger="collectra"):
        spy(tmp_path, keep=1).run(files=[done], force=True)
    assert outside.read_text() == "shared by several people"
    assert [i["data"] for i in record(tmp_path, "parts")] == ["new 1"]
    assert any("outside" in r.getMessage() for r in caplog.records)


def test_image_artefacts_never_lose_their_source_files(tmp_path, files):
    pipe = spy(tmp_path)
    pipe.connect()
    node = pipe.node_manager.resolve_node("spy")
    output = pipe.node_manager.resolve_node("parts")
    source = Path(files[0])
    output.add_item(Image(name="parts", id="parts", data=source))
    results = tmp_path / "results"
    results.mkdir()
    previous = pipe._take_outputs(node, results)
    assert source.exists()
    assert output.items == {}
    previous.restore()
    assert list(output.items) == ["parts"]


def test_artefacts_say_which_files_they_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Text(name="t", id="t", data="x").stored_files(tmp_path) == []
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.md").write_text("x")
    stored = Text(name="t", id="t", path="a/b.md")
    assert stored.stored_files(tmp_path) == [tmp_path / "a" / "b.md"]
