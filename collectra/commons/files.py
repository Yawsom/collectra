import uuid
from pathlib import Path

import pydantic

from ..types.base import load_artefact
from ..utils import write_yaml
from .results import Results, pop_header, put_header

# The header used to be called this; the name still works.
CollectraResultsMetadata = Results


class CollectraFile(pydantic.BaseModel):
    collectra_file_path: Path
    collectra_results_metadata: CollectraResultsMetadata
    data: dict
    assets: dict[str, Path] = dict()
    cleanup_on_save_failure: bool = False

    def __getitem__(self, key):
        item = self.data[key]
        return item

    def __contains__(self, key):
        return key in self.data

    @property
    def results_path(self) -> Path:
        """Return the canonical results file for this Collectra directory."""
        return self.collectra_file_path / "results.yaml"

    def model_dump(self, *args, **kwargs) -> dict:
        """What is written to results.yaml: the header, then each node."""
        data = super().model_dump(*args, **kwargs)["data"]
        nodes = {
            key: (
                value
                if not isinstance(value, list)
                else value[0] if len(value) == 1 else value
            )
            for key, value in data.items()
        }
        return put_header(self.collectra_results_metadata, nodes)

    @classmethod
    def from_data(cls, collectra_file_path: Path | str):
        import yaml

        collectra_file_path = Path(collectra_file_path)
        results_yaml = collectra_file_path / "results.yaml"
        if not results_yaml.exists():
            raise FileNotFoundError(f"results.yaml not found in {collectra_file_path}")
        with open(results_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid results data in {results_yaml}")
        metadata = pop_header(data, source=results_yaml)
        return cls(
            collectra_file_path=collectra_file_path,
            collectra_results_metadata=metadata,
            data=data,
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        label: str,
        ext: str,
        output: Path | None = None,
        force: bool = False,
        **kwargs,
    ):
        ext = ext.lstrip(".")
        collectra_file_path = file.with_suffix(f".{ext}")
        if output:
            collectra_file_path = output / collectra_file_path.name
        directory_existed = collectra_file_path.exists()
        if directory_existed and not force:
            raise FileExistsError(
                f"File {collectra_file_path} already exists. To overwrite, use -f or --force option."
            )
        else:
            collectra_file_path.mkdir(parents=True, exist_ok=True)
        partition = kwargs.get("partition", None)
        metadata = CollectraResultsMetadata(partition=partition)
        new_data = {
            f"{label}": {
                "id": f"{label}",
                "type": "collectra.Image",
                "data": str(file.name),
            }
        }
        return cls(
            collectra_file_path=collectra_file_path,
            collectra_results_metadata=metadata,
            data=new_data,
            assets={file.name: Path(file)},
            cleanup_on_save_failure=not directory_existed,
        )

    def save(self):
        import copy
        import shutil

        collectra_file_path = self.collectra_file_path
        assets = copy.deepcopy(self.assets)
        data = self.model_dump()
        try:
            for asset_name, asset_path in assets.items():
                if not asset_path.exists():
                    raise FileNotFoundError(f"Asset file {asset_path} not found.")
                shutil.copy(asset_path, collectra_file_path / asset_name)
            write_yaml(data, self.results_path)
            self.cleanup_on_save_failure = False
        except Exception as e:
            if self.cleanup_on_save_failure:
                shutil.rmtree(collectra_file_path, ignore_errors=True)
            raise RuntimeError(f"Failed to save Collectra file: {e}")
