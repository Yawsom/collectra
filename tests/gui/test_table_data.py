import yaml


def make_page(root, name, texts):
    folder = root / f"{name}.collectra"
    folder.mkdir()
    data = {}
    for label, values in texts.items():
        data[label] = [
            {"type": "collectra.Text", "id": f"{label}_{i}", "data": value}
            for i, value in enumerate(values)
        ]
    data["picture"] = [{"type": "collectra.Image", "id": "img", "data": "x.jpg"}]
    (folder / "results.yaml").write_text(yaml.dump(data))
    return {
        "name": folder.name,
        "path": str(folder),
        "yaml_path": str(folder / "results.yaml"),
        "image_path": "",
    }


class TestGetTableData:
    def test_no_folders_loaded(self, backend_with_extension):
        assert backend_with_extension.get_table_data() == {
            "success": True,
            "columns": [],
            "rows": [],
        }

    def test_one_row_per_page_with_text_columns_only(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        backend._collectra_folders = [
            make_page(tmp_path, "a", {"title": ["Alpha"], "notes": ["one", "two"]}),
            make_page(tmp_path, "b", {"title": ["Beta"]}),
        ]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title", "notes"])

        result = backend.get_table_data()

        assert result["success"] is True
        assert result["columns"] == ["title", "notes"]  # no image column
        assert result["rows"] == [
            {
                "index": 0,
                "name": "a.collectra",
                "title": "Alpha",
                "notes": "one | two",
                "__editable": ["title"],  # "notes" has two artefacts
            },
            {"index": 1, "name": "b.collectra", "title": "Beta", "__editable": ["title"]},
        ]

    def test_labels_missing_from_the_pipeline_are_added(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        backend._collectra_folders = [make_page(tmp_path, "a", {"extra": ["hand made"]})]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title"])

        assert backend.get_table_data()["columns"] == ["title", "extra"]

    def test_long_text_is_cut(self, backend_with_extension, tmp_path, monkeypatch):
        backend = backend_with_extension
        backend._collectra_folders = [make_page(tmp_path, "a", {"title": ["x" * 50]})]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title"])

        cell = backend.get_table_data(max_chars=10)["rows"][0]["title"]

        assert cell == "x" * 10 + "…"

    def test_an_unreadable_page_is_still_a_row(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        page = make_page(tmp_path, "a", {"title": ["Alpha"]})
        (tmp_path / "a.collectra" / "results.yaml").write_text("{{ not yaml")
        backend._collectra_folders = [page, make_page(tmp_path, "b", {"title": ["Beta"]})]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title"])

        rows = backend.get_table_data()["rows"]

        assert rows[0] == {"index": 0, "name": "a.collectra", "__editable": []}
        assert rows[1]["title"] == "Beta"

    def test_columns_follow_the_pipeline_text_nodes(self):
        from pathlib import Path

        from collectra.gui.backend import GUIBackend
        from collectra.pipelines.base import Pipeline

        pipeline = Pipeline.from_file(
            Path(__file__).parents[1] / "data" / "collectrapipeline" / "pipeline.yaml"
        )

        labels = GUIBackend(pipeline)._text_artefact_labels()

        assert "md_example" in labels  # a collectra.Text
        assert "bar_img" not in labels  # an image
        assert "human" not in labels  # an image crop


class TestSetTableCells:
    def pages(self, backend, tmp_path, monkeypatch):
        backend._collectra_folders = [
            make_page(tmp_path, "a", {"title": ["Alpha"], "notes": ["one", "two"]}),
            make_page(tmp_path, "b", {"title": ["Beta"]}),
        ]

    def test_saves_to_every_page_and_skips_what_cannot_be_set(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        self.pages(backend, tmp_path, monkeypatch)

        result = backend.set_table_cells(
            [
                {"index": 0, "label": "title", "text": "New"},
                {"index": 1, "label": "title", "text": "New"},
                {"index": 0, "label": "notes", "text": "New"},  # two artefacts
                {"index": 1, "label": "notes", "text": "New"},  # none
                {"index": 9, "label": "title", "text": "New"},  # no such page
            ]
        )

        assert [(s["index"], s["label"]) for s in result["saved"]] == [(0, "title"), (1, "title")]
        assert sorted((s["index"], s["label"]) for s in result["skipped"]) == [
            (0, "notes"), (1, "notes"), (9, "title"),
        ]
        rows = backend.get_table_data()["rows"]
        assert [row["title"] for row in rows] == ["New", "New"]
        assert rows[0]["notes"] == "one | two"  # untouched

    def test_saves_to_the_page_that_is_open(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        self.pages(backend, tmp_path, monkeypatch)
        assert backend.load_yaml(backend._collectra_folders[0]["yaml_path"])["success"]

        result = backend.set_table_cells([{"index": 0, "label": "title", "text": "Open"}])

        assert len(result["saved"]) == 1
        assert backend.get_table_data()["rows"][0]["title"] == "Open"
        # ...and it is on disk, not only in the open page
        assert "Open" in (tmp_path / "a.collectra" / "results.yaml").read_text()


class TestExportTableCsv:
    def test_writes_the_table_to_the_chosen_file(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        backend = backend_with_extension
        backend._collectra_folders = [
            make_page(tmp_path, "a", {"title": ["Alpha, \"one\"\nline"], "notes": ["x", "y"]}),
            make_page(tmp_path, "b", {"title": ["Beta"]}),
        ]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title", "notes"])
        target = tmp_path / "out.csv"
        asked = {}

        class Window:
            def create_file_dialog(self, **kwargs):
                asked.update(kwargs)
                return str(target)

        backend._window = Window()

        result = backend.export_table_csv({"title": "Title"})

        assert result == {"success": True, "path": str(target)}
        assert asked["save_filename"] == "table.csv"
        import csv

        with open(target, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows == [
            ["File", "Title", "notes"],
            ["a.collectra", "Alpha, \"one\"\nline", "x | y"],
            ["b.collectra", "Beta", ""],
        ]

    def test_cancelled_dialog_writes_nothing(self, backend_with_extension, tmp_path, monkeypatch):
        backend = backend_with_extension
        backend._collectra_folders = [make_page(tmp_path, "a", {"title": ["A"]})]
        monkeypatch.setattr(backend, "_text_artefact_labels", lambda: ["title"])
        backend._window = type("W", (), {"create_file_dialog": lambda self, **k: None})()

        assert backend.export_table_csv() == {"success": False, "cancelled": True}
