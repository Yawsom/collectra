"""scripts/upgrade_results_files.py moves old results headers to the new form."""

import importlib.util
import json
import warnings
from pathlib import Path

import pytest
import yaml

from collectra.commons.files import CollectraFile
from collectra.commons.results import HEADER_KEY, LEGACY_KEY, upgrade_header

SCRIPT = Path(__file__).parents[2] / "scripts" / "upgrade_results_files.py"
spec = importlib.util.spec_from_file_location("upgrade_results_files", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

OLD = {"workflow": "W", "version": "1", "partition": "train", "timestamp": "2026-01-01"}
IMAGE = {"type": "collectra.Image", "id": "page", "data": "page.png"}


def write(folder: Path, data: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def load(folder: Path) -> dict:
    return yaml.safe_load((folder / "results.yaml").read_text())


# --- the conversion ---------------------------------------------------------------------


def test_the_old_header_becomes_the_first_entry_keeping_its_values():
    upgraded = upgrade_header({LEGACY_KEY: dict(OLD), "page": dict(IMAGE)})
    assert list(upgraded) == [HEADER_KEY, "page"]
    assert upgraded[HEADER_KEY] == {
        "type": "collectra.Results",
        **OLD,
    }  # timestamp kept
    assert upgraded["page"] == IMAGE


def test_an_empty_old_header_still_becomes_an_entry():
    assert upgrade_header({LEGACY_KEY: None, "page": dict(IMAGE)})[HEADER_KEY] == {
        "type": "collectra.Results"
    }


def test_a_node_called_results_gets_the_header_under_another_key():
    upgraded = upgrade_header({LEGACY_KEY: dict(OLD), "results": dict(IMAGE)})
    assert list(upgraded) == ["collectra_results", "results"]


@pytest.mark.parametrize(
    "data",
    [
        {HEADER_KEY: {"type": "collectra.Results"}, "page": dict(IMAGE)},
        {"page": dict(IMAGE)},
    ],
)
def test_nothing_to_upgrade_gives_none(data):
    assert upgrade_header(data) is None


# --- the script -------------------------------------------------------------------------


def run(args, capsys):
    code = script.main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_a_file_is_upgraded_and_then_loads_without_a_warning(tmp_path, capsys):
    folder = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD), "page": dict(IMAGE)})
    code, out = run([folder], capsys)
    assert code == 0 and "Upgraded 1 of 1" in out
    assert list(load(folder)) == [HEADER_KEY, "page"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        file = CollectraFile.from_data(folder)
    assert file.collectra_results_metadata.partition == "train"


def test_dry_run_changes_nothing(tmp_path, capsys):
    folder = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD), "page": dict(IMAGE)})
    before = (folder / "results.yaml").read_text()
    code, out = run([folder, "--dry-run"], capsys)
    assert code == 0 and "Would upgrade 1 of 1" in out
    assert (folder / "results.yaml").read_text() == before


def test_running_again_finds_nothing_to_do(tmp_path, capsys):
    folder = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD), "page": dict(IMAGE)})
    run([folder], capsys)
    after = (folder / "results.yaml").read_text()
    code, out = run([folder], capsys)
    assert code == 0 and "Nothing to upgrade in 1" in out
    assert (folder / "results.yaml").read_text() == after


def test_a_parent_folder_is_searched_and_other_files_are_left_alone(tmp_path, capsys):
    old = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD), "page": dict(IMAGE)})
    new = write(tmp_path / "b.x", {HEADER_KEY: {"type": "collectra.Results"}})
    bare = write(tmp_path / "c.x", {"page": dict(IMAGE)})
    texts = {p: (p / "results.yaml").read_text() for p in (new, bare)}
    code, out = run([tmp_path], capsys)
    assert "Upgraded 1 of 3" in out
    assert list(load(old))[0] == HEADER_KEY
    assert {p: (p / "results.yaml").read_text() for p in (new, bare)} == texts


def test_a_results_yaml_path_can_be_given_directly(tmp_path, capsys):
    folder = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD)})
    run([folder / "results.yaml"], capsys)
    assert list(load(folder)) == [HEADER_KEY]


def test_a_file_that_is_not_a_mapping_is_skipped(tmp_path, capsys):
    folder = tmp_path / "a.x"
    folder.mkdir()
    (folder / "results.yaml").write_text("- just\n- a list\n")
    code, out = run([folder], capsys)
    assert code == 0 and "Nothing to upgrade" in out


def test_no_results_files_is_reported(tmp_path, capsys):
    assert script.main([str(tmp_path)]) == 2
    assert "No results.yaml files found" in capsys.readouterr().err


def test_json_output_lists_the_files(tmp_path, capsys):
    folder = write(tmp_path / "a.x", {LEGACY_KEY: dict(OLD)})
    code, out = run([folder, "--json", "--dry-run"], capsys)
    (entry,) = json.loads(out)
    assert code == 0 and entry["file"].endswith("results.yaml")
    assert entry["applied"] is False
