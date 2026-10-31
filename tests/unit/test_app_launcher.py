"""Validate app bundles and execute their launcher without opening a GUI."""

import json
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from collectra.cli import invoke
from collectra.pipelines.app_launcher import install_app
from collectra.pipelines.base import Pipeline


@pytest.fixture
def pipeline_file(tmp_path):
    directory = tmp_path / "pipeline with 'quotes' and $symbols"
    directory.mkdir()
    path = directory / "pipeline.yaml"
    path.write_text(
        "collectra_pipeline_metadata: {name: test, ext: collectra, version: '1.0'}\n"
    )
    return path


def test_cli_bundle_and_launch_arguments(tmp_path, pipeline_file, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    capture = tmp_path / "captured.json"
    interpreter = tmp_path / "python with 'quotes'"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f'with open({str(capture)!r}, "w") as stream:\n'
        '    json.dump({"args": sys.argv[1:], "cwd": os.getcwd()}, stream)\n'
    )
    interpreter.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    inputs = tmp_path / "results with spaces"
    inputs.mkdir()
    apps = tmp_path / "Applications"
    pipeline = Pipeline.from_file(pipeline_file)
    invoke(
        pipeline,
        ["install-app", str(inputs), "--name", "My Pipeline", "--app-dir", str(apps)],
    )
    bundle = apps / "My Pipeline.app"
    with (bundle / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    assert info["CFBundlePackageType"] == "APPL"
    assert info["CFBundleDisplayName"] == "My Pipeline"
    assert (bundle / "Contents/Resources/AppIcon.icns").is_file()
    launcher = bundle / "Contents/MacOS" / info["CFBundleExecutable"]
    subprocess.run([str(launcher)], cwd=tmp_path, check=True)
    captured = json.loads(capture.read_text())
    assert captured["cwd"] == str(pipeline_file.parent.resolve())
    assert captured["args"] == [
        "-m",
        "collectra.main",
        "--pipeline",
        str(pipeline_file.resolve()),
        "gui",
        str(inputs.resolve()),
    ]
    with pytest.raises(FileExistsError):
        install_app(pipeline_file, name="My Pipeline", app_dir=apps)
    assert launcher.exists()


def test_defaults_and_custom_icon(tmp_path, pipeline_file, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    icon = tmp_path / "custom.icns"
    icon.write_bytes(b"custom icon")
    bundle = install_app(pipeline_file, icon=icon)
    assert bundle == tmp_path / "Applications/Collectra.app"
    assert (bundle / "Contents/Resources/AppIcon.icns").read_bytes() == b"custom icon"


@pytest.mark.parametrize("architecture", ["arm64", "x86_64"])
def test_app_pins_installing_python_architecture(tmp_path, pipeline_file, monkeypatch, architecture):
    import shlex
    from collectra.pipelines import app_launcher

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(app_launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(app_launcher.platform, "machine", lambda: architecture)
    monkeypatch.setattr(sys, "executable", "/Python Environment/bin/python")
    bundle = install_app(pipeline_file, app_dir=tmp_path)
    lines = (bundle / "Contents/MacOS/collectra").read_text().splitlines()
    assert shlex.split(lines[-1])[:4] == [
        "exec", "/usr/bin/arch", f"-{architecture}", "/Python Environment/bin/python",
    ]


@pytest.mark.parametrize("name", ["", " ", "..", "../bad", "bad:name", "bad\\name"])
def test_invalid_name(tmp_path, pipeline_file, monkeypatch, name):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError):
        install_app(pipeline_file, name=name, app_dir=tmp_path)


def test_platform_and_icon_validation(tmp_path, pipeline_file, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        install_app(pipeline_file, app_dir=tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match=".icns"):
        install_app(pipeline_file, app_dir=tmp_path, icon=tmp_path / "missing.icns")


@pytest.mark.parametrize("metadata_key", ["icon", "logo"])
def test_pipeline_icon_and_explicit_override(
    tmp_path, pipeline_file, monkeypatch, metadata_key
):
    from PIL import Image

    monkeypatch.setattr(sys, "platform", "darwin")
    image_path = pipeline_file.parent / "pipeline.png"
    Image.new("RGBA", (128, 128), "red").save(image_path)
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.pipeline_metadata[metadata_key] = "pipeline.png"
    bundle = pipeline.cli_install_app(name="Pipeline Icon", app_dir=tmp_path)
    icon_path = bundle / "Contents/Resources/AppIcon.icns"
    with Image.open(icon_path) as icon:
        assert icon.format == "ICNS"
        assert icon.convert("RGBA").getpixel((0, 0)) == (255, 0, 0, 255)
    override = tmp_path / "override.icns"
    override.write_bytes(b"override icon")
    bundle = pipeline.cli_install_app(name="Override", app_dir=tmp_path, icon=override)
    assert (bundle / "Contents/Resources/AppIcon.icns").read_bytes() == b"override icon"


def test_force_replaces_bundle_via_cli(tmp_path, pipeline_file, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    pipeline = Pipeline.from_file(pipeline_file)
    bundle = install_app(pipeline_file, app_dir=tmp_path)
    old_file = bundle / "old-file"
    old_file.write_text("old")
    invoke(pipeline, ["install-app", "--app-dir", str(tmp_path), "--force"])
    assert not old_file.exists()
    assert (bundle / "Contents/Info.plist").is_file()
    assert not list(tmp_path.glob(".collectra-app-*"))


def test_failed_forced_install_preserves_existing_app(
    tmp_path, pipeline_file, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "darwin")
    bundle = install_app(pipeline_file, app_dir=tmp_path)
    old_file = bundle / "keep"
    old_file.write_text("keep")
    invalid_icon = tmp_path / "invalid.png"
    invalid_icon.write_bytes(b"not a PNG")
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        install_app(pipeline_file, app_dir=tmp_path, icon=invalid_icon, force=True)
    assert old_file.read_text() == "keep"
    assert not list(tmp_path.glob(".collectra-app-*"))
