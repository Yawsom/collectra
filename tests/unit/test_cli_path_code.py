"""``path`` echoes the pipeline folder; ``code`` opens it in VS Code when installed."""

import subprocess

import cappa
import pytest

from collectra import Collectra
from collectra.cli import command, invoke


def pipeline(tmp_path):
    folder = tmp_path / "My Pipeline.collectra"
    folder.mkdir()
    return Collectra("Demo", "dmo", "0.1.0", path=str(folder)), folder


def root_help(app, capsys):
    with pytest.raises(cappa.HelpExit):
        invoke(app, ["--help"], name="livy")
    return capsys.readouterr().out


def test_path_echoes_the_pipeline_folder_and_nothing_else(tmp_path, capsys):
    app, folder = pipeline(tmp_path)
    invoke(app, ["path"], name="livy")
    assert capsys.readouterr().out == f"{folder.resolve()}\n"


def test_path_is_listed_in_help(tmp_path, capsys):
    app, _ = pipeline(tmp_path)
    assert "path" in root_help(app, capsys)


def test_code_opens_the_pipeline_folder_in_vs_code(tmp_path, monkeypatch):
    app, folder = pipeline(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/code")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 0),
    )
    invoke(app, ["code"], name="livy")
    assert calls == [["/usr/local/bin/code", str(folder.resolve())]]


def test_code_is_offered_only_when_vs_code_is_installed(tmp_path, capsys, monkeypatch):
    app, _ = pipeline(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert "code" not in root_help(app, capsys).replace("Code", "")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/code")
    assert "code" in root_help(app, capsys)


def test_code_reports_a_failed_launch(tmp_path, monkeypatch):
    app, _ = pipeline(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/code")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )
    with pytest.raises(SystemExit, match="VS Code"):
        invoke(app, ["code"], name="livy")


def test_commands_can_declare_when_they_are_available():
    class Tool:
        @command(name="always")
        def always(self) -> None:
            """Always there."""

        @command(name="never", available=lambda: False)
        def never(self) -> None:
            """Never there."""

    from collectra.cli import command_for_object

    names = {
        option.name
        for option in command_for_object(Tool(), name="tool")
        .arguments[0]
        .options.values()
    }
    assert names == {"always"}
