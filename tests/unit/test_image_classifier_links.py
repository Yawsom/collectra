from types import SimpleNamespace

import pytest
import yaml

from collectra import Collectra, Image, ImageClassifierYOLO, Link
from collectra.types.base import ArtefactNode


def make_image(tmp_path, name="Palynomorph", filename="image.jpg"):
    from PIL import Image as PillowImage

    path = tmp_path / filename
    PillowImage.new("RGB", (12, 12), "white").save(path)
    return Image(name=name, id=f"{name}15", data=path)


def test_classifier_returns_link_to_input_image(tmp_path):
    image = make_image(tmp_path)
    task = ImageClassifierYOLO(
        "classifier",
        model="unused.pt",
        output=["Algae", "Fungi", "Pollen"],
    )
    task._init_model = lambda: None
    task.model = lambda _: [
        SimpleNamespace(
            probs=SimpleNamespace(top1=1),
            names={0: "Algae", 1: "Fungi", 2: "Pollen"},
        )
    ]

    result = task.run(image)
    result.set_parents([image])

    assert type(result) is Link
    assert isinstance(result, Image)
    assert result.name == "Fungi"
    assert result.target is image
    assert result.get_path() == image.get_path()
    assert result.serialize() == {
        "type": "collectra.Link",
        "id": result.id,
        "parents": "Palynomorph15",
    }


def test_classifier_output_nodes_are_typed_as_links(tmp_path):
    make_image(tmp_path)
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "collectra_pipeline_metadata": {
                    "name": "Palynomorph",
                    "ext": "palynomorph",
                    "version": "1.0",
                },
                "Palynomorph": {
                    "type": "collectra.Image",
                    "data": "image.jpg",
                },
                "palynomorph_classifier": {
                    "type": "collectra.ImageClassifierYOLO",
                    "model": "yolo11x.pt",
                    "input": "Palynomorph",
                    "output": ["Algae", "Fungi", "Pollen"],
                },
            },
            sort_keys=False,
        )
    )
    pipeline = Collectra.from_file(pipeline_file)
    pipeline.connect()

    for name in ("Algae", "Fungi", "Pollen"):
        node = pipeline.node_manager.resolve_node(name)
        assert isinstance(node, ArtefactNode)
        assert node.types == {Link}


def test_link_loaded_from_yaml_can_bind_to_first_parent(tmp_path):
    image = make_image(tmp_path)
    result_dir = tmp_path / "sample.palynomorph"
    result_dir.mkdir()
    (result_dir / "results.yaml").write_text(
        yaml.safe_dump(
            {
                "collectra_results_metadata": {},
                "Fungi": {
                    "type": "collectra.Link",
                    "id": "Fungi1",
                    "parents": image.id,
                },
            }
        )
    )
    node = ArtefactNode("Fungi", types={Link})
    node.process("Fungi", result_dir)
    link = next(iter(node.items.values()))

    assert type(link) is Link
    assert link.target is None
    assert link.bind({image.id: image}) is image
    assert isinstance(link, Image)


def test_classifier_training_uses_link_node_name_as_class(tmp_path):
    image = make_image(tmp_path)
    link = Link(name="Fungi", id="Fungi1", parents=[image.id])
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    training = task.prepare_training_inputs(
        [],
        [image],
        {"sample": [link]},
        {"sample": [image]},
    )

    assert training == [link]
    assert link.target is image
    assert isinstance(link, Image)
    assert link.name == "Fungi"


def test_link_resolves_through_multiple_links(tmp_path):
    image = make_image(tmp_path)
    fungi = Link(name="Fungi", id="Fungi1", parents=[image.id], target=image)
    bulbilspore = Link(
        name="Bulbilspore", id="Bulbilspore1", parents=[fungi.id], target=fungi
    )

    assert bulbilspore.resolve() is image
    assert isinstance(bulbilspore, Image)
    assert bulbilspore.get_path() == image.get_path()


def test_link_reports_cycles():
    first = Link(name="First", id="First1", parents=["Second1"])
    second = Link(name="Second", id="Second1", parents=["First1"])
    first.target = second
    second.target = first

    with pytest.raises(RuntimeError, match="Cyclic Link chain.*First1.*Second1"):
        first.resolve()


def test_classifier_training_resolves_link_to_link(tmp_path):
    image = make_image(tmp_path)
    image.partition = "training"
    # YAML's compact ``parents: id`` form deserializes as a scalar string.
    fungi = Link(name="Fungi", id="Fungi1", parents=image.id)
    bulbilspore = Link(name="Bulbilspore", id="Bulbilspore1", parents=fungi.id)
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    training = task.prepare_training_inputs(
        [],
        [fungi, image],
        {"sample": [bulbilspore]},
        {"sample": [fungi, image]},
    )

    assert training == [bulbilspore]
    assert bulbilspore.target is fungi
    assert fungi.target is image
    assert bulbilspore.resolve() is image
    assert bulbilspore.partition == "training"
    assert fungi.parent_id == image.id
    assert bulbilspore.parent_id == fungi.id


def test_classifier_training_reports_missing_link_parent(tmp_path):
    image = make_image(tmp_path)
    link = Link(name="Fungi", id="Fungi1", parents=["missing-parent"])
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    with pytest.raises(ValueError, match="Fungi1.*missing parent.*missing-parent"):
        task.prepare_training_inputs(
            [],
            [image],
            {"sample": [link]},
            {"sample": [image]},
        )


def test_image_crop_reports_empty_pixel_geometry(tmp_path):
    image = make_image(tmp_path)
    crop = image.make_crop(
        x_center=0.5,
        y_center=0.5,
        width_relative=0.0,
        height_relative=0.25,
        name="Palynomorph",
    )
    crop.source_file = tmp_path / "invalid.palynomorph"

    with pytest.raises(ValueError) as error:
        crop.pil()

    message = str(error.value)
    assert crop.id in message
    assert "invalid.palynomorph" in message
    assert str(image.get_path()) in message
    assert "width=0.0" in message


def test_unclassified_file_is_ignored_for_link_training(tmp_path):
    image = make_image(tmp_path)
    result_dir = tmp_path / "unclassified.palynomorph"
    result_dir.mkdir()
    (result_dir / "results.yaml").write_text(
        yaml.safe_dump(
            {
                "collectra_results_metadata": {"partition": "training"},
                "image": {
                    "type": "collectra.Image",
                    "id": "image",
                    "data": str(image.get_path()),
                },
                "Palynomorph": {
                    "type": "collectra.ImageCrop",
                    "id": "Palynomorph1",
                    "parents": "image",
                    "data": str(image.get_path()),
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width_relative": 0.1,
                    "height_relative": 0.1,
                },
            }
        )
    )
    output_nodes = [
        ArtefactNode(name, types={Link})
        for name in ("Algae", "Dinocyst", "Fungi", "Pollen", "Spore", "TCT")
    ]

    assert ArtefactNode.batch_process(result_dir, output_nodes) == []

    negative_samples = ArtefactNode.batch_process(
        result_dir,
        output_nodes,
        include_unlabelled=True,
    )
    assert len(negative_samples) == 1
    assert isinstance(negative_samples[0], Image)


def test_classifier_skips_empty_linked_crop_with_ids(tmp_path, caplog):
    image = make_image(tmp_path)
    crop = image.make_crop(
        x_center=0.5,
        y_center=0.0,
        width_relative=0.25,
        height_relative=0.0,
        name="Palynomorph",
    )
    crop.id = "Palynomorph1"
    crop.source_file = tmp_path / "sample.palynomorph"
    link = Link(
        name="Pollen",
        id="Pollen1",
        parents=[crop.id],
        target=crop,
        partition="training",
    )
    link.source_file = crop.source_file
    task = ImageClassifierYOLO("classifier", model="unused.pt")
    log = tmp_path / "training"

    train, validation = task._prepare_assets(
        ["Pollen"],
        log,
        "validation",
        "",
        link,
    )

    assert train == log / "train"
    assert validation == log / "val"
    assert list(train.rglob("*.jpg")) == []
    assert "Pollen1 -> Palynomorph1" in caplog.text
    assert "sample.palynomorph" in caplog.text
    assert "image is 3x0 pixels" in caplog.text


def test_classifier_writes_each_linked_crop_with_a_unique_name(tmp_path, monkeypatch):
    image = make_image(tmp_path)
    links = []
    for index, x_center in enumerate((0.25, 0.75), start=1):
        crop = image.make_crop(
            x_center=x_center,
            y_center=0.5,
            width_relative=0.25,
            height_relative=0.25,
            name="Palynomorph",
        )
        crop.id = f"Palynomorph{index}"
        links.append(
            Link(
                name="Pollen",
                id=f"Pollen{index}",
                parents=[crop.id],
                target=crop,
                partition="training",
            )
        )
    task = ImageClassifierYOLO("classifier", model="unused.pt")
    log = tmp_path / "training"
    from collectra.types.images import open_oriented as real_open

    opened = []

    def tracking_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(
        "collectra.tasks.image_classifier.yolo.open_oriented", tracking_open
    )

    train, _ = task._prepare_assets(["Pollen"], log, "validation", "", *links)

    assert sorted(path.name for path in train.rglob("*.jpg")) == [
        f"{image.get_path().stem}-Pollen1.jpg",
        f"{image.get_path().stem}-Pollen2.jpg",
    ]
    assert opened == [image.get_path().resolve()]


def test_classifier_min_size_also_applies_to_whole_images(tmp_path, caplog):
    from PIL import Image as PillowImage

    path = tmp_path / "tiny.jpg"
    PillowImage.new("RGB", (1, 10), "white").save(path)
    image = Image(name="specimen", id="image1", data=path)
    link = Link(
        name="Pollen",
        id="Pollen1",
        parents=[image.id],
        target=image,
        partition="training",
    )
    task = ImageClassifierYOLO("classifier", model="unused.pt")
    log = tmp_path / "training"

    train, _ = task._prepare_assets(["Pollen"], log, "validation", "", link, min_size=2)

    assert list(train.rglob("*.jpg")) == []
    assert "Pollen1 -> image1" in caplog.text
    assert "image is 1x10 pixels (minimum 2)" in caplog.text


def test_classifier_min_size_zero_disables_filtering(tmp_path):
    from PIL import Image as PillowImage

    path = tmp_path / "tiny.jpg"
    PillowImage.new("RGB", (1, 1), "white").save(path)
    image = Image(name="specimen", id="image1", data=path)
    link = Link(
        name="Pollen",
        id="Pollen1",
        parents=[image.id],
        target=image,
        partition="training",
    )
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    train, _ = task._prepare_assets(
        ["Pollen"], tmp_path / "training", "validation", "", link
    )

    assert [path.name for path in train.rglob("*.jpg")] == ["tiny-Pollen1.jpg"]


def test_classifier_rejects_negative_min_size(tmp_path):
    task = ImageClassifierYOLO("classifier", model="unused.pt")

    with pytest.raises(ValueError, match="min_size cannot be negative"):
        task._prepare_assets(
            ["Pollen"], tmp_path / "training", "validation", "", min_size=-1
        )


def test_classification_distribution_includes_totals(tmp_path, capsys):
    for partition, class_name, count in (
        ("train", "Pollen", 2),
        ("train", "Spore", 1),
        ("val", "Pollen", 1),
        ("val", "Spore", 3),
    ):
        directory = tmp_path / partition / class_name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{index}.jpg").touch()

    task = ImageClassifierYOLO("classifier", model="unused.pt")
    task._check_distribution(tmp_path, ["Pollen", "Spore"])

    output = capsys.readouterr().out
    assert "Validation %" in output
    total_row = next(line for line in output.splitlines() if "Total" in line)
    assert "3" in total_row
    assert "4" in total_row
    assert "57.1%" in total_row
