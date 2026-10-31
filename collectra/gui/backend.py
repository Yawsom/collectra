"""
API backend for annotation display logic via pywebview's js_api interface.

JavaScript usage:
    await pywebview.api.load_yaml("path/to/results.yaml")
    const result = await pywebview.api.get_display_value("node_id")
    const nodes = await pywebview.api.get_all_nodes()
"""

import base64
import csv
import enum
import platform
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import webview

from collectra.commons.files import CollectraFile

from .scss_theme import compile_theme
from .data_display import CollectraAnnotationNode, CollectraGraph, NodeDisplayValue

if TYPE_CHECKING:
    from collectra.pipelines.base import Pipeline


class ImageFormat(enum.Enum):
    PNG = "image/png"
    JPEG = "image/jpeg"
    JPG = "image/jpg"
    GIF = "image/gif"
    BMP = "image/bmp"
    WEBP = "image/webp"
    TIFF = "image/tiff"
    TIF = "image/tiff"


class GUIBackend:
    """
    API class exposed to JavaScript via pywebview.

    All public methods are accessible in JS as:
        window.pywebview.api.methodName(args)
    """

    def __init__(self, pipeline: "Pipeline", inputs=(), max_pixels: int = 4096):
        """
        Initialize the API instance.

        Sets up the annotation graph, YAML path, and window reference.
        All values are initially None until load_yaml() is called.
        """
        if type(max_pixels) is not int or max_pixels < 1:
            raise ValueError("max_pixels must be a positive integer")
        self._max_pixels = max_pixels
        self._git_lock = threading.Lock()
        self._text_save_lock = threading.Lock()
        self._update_lock = threading.Lock()
        self._restart_requested = False
        self._restart_required = False
        self._pipeline = pipeline
        self._collectra_file: CollectraFile | None = None
        self._graph: CollectraGraph | None = None
        self._yaml_path: str | None = None
        self._window = None
        self._collectra_folders: list[dict] = []
        self._parent_folder: str | None = None
        self._global_labels: set[str] = set()
        self._global_label_counts: dict[str, int] = {}  # {label: total_count}
        from .session_cache import FolderSessionCache

        pipeline_path = getattr(pipeline, "path", None)
        self._session_cache = (
            FolderSessionCache(pipeline_path) if pipeline_path else None
        )
        from .arrangement import ArrangementStore

        metadata = getattr(pipeline, "pipeline_metadata", {})
        layouts = metadata.get("layouts")
        default_layout = None
        if pipeline_path and isinstance(layouts, list) and layouts:
            if isinstance(layouts[0], str) and layouts[0].strip():
                default_layout = Path(pipeline_path) / layouts[0]
        self._arrangement = (
            ArrangementStore(
                pipeline_path,
                pipeline_name=metadata.get("name") or getattr(pipeline, "name", None),
                default_layout=default_layout,
            )
            if pipeline_path
            else None
        )
        self._initial_items = (
            self._load_input_paths(inputs) if inputs else self._restore_folder_session()
        )

    def _save_folder_session(self, folders=None) -> None:
        if self._session_cache is not None:
            paths = (
                folders
                if folders is not None
                else [folder["path"] for folder in self._collectra_folders]
            )
            self._session_cache.write([str(Path(path).resolve()) for path in paths])

    def _restore_folder_session(self):
        if self._session_cache is None:
            return None
        self._collectra_folders = []
        seen = set()
        for value in self._session_cache.read():
            try:
                path = Path(value).resolve()
                if path in seen or not path.is_dir():
                    continue
                scan = self._scan_collectra_folder(path)
                if scan is None:
                    continue
                seen.add(path)
                self._collectra_folders.append(
                    {"name": path.name, "path": str(path), **scan}
                )
            except (OSError, ValueError):
                continue
        self._save_folder_session()
        if not self._collectra_folders:
            return None
        self._parent_folder = str(Path(self._collectra_folders[0]["path"]).parent)
        active = self._session_cache.read_active()
        active_index = next(
            (
                index
                for index, folder in enumerate(self._collectra_folders)
                if folder["path"] == active
            ),
            0,
        )
        return {
            **self._folder_list_result(),
            "provided": True,
            "mode": "parent",
            "active_index": active_index,
        }

    def git_action(
        self, action: str, message: str = "", repository: str = "", strategy: str = ""
    ) -> dict:
        """Operate on the Git repository containing the active results file."""
        from .source_control import run_git

        if not self._yaml_path:
            return {"success": False, "error": "Open a results folder first"}
        if not self._git_lock.acquire(blocking=False):
            return {"success": False, "error": "A Git operation is already running"}
        try:
            return run_git(
                Path(self._yaml_path).parent, action, message, repository, strategy
            )
        finally:
            self._git_lock.release()

    def get_arrangement(self) -> dict:
        """The saved panel/tab arrangement for this pipeline, or None."""
        if self._arrangement is None:
            return {"success": True, "arrangement": None}
        return {"success": True, "arrangement": self._arrangement.read()}

    def save_arrangement(self, arrangement: dict) -> dict:
        """Remember the panel/tab arrangement (per user, per pipeline)."""
        from .arrangement import sanitize_arrangement

        clean = sanitize_arrangement(arrangement)
        if clean is None:
            return {"success": False, "error": "Not a valid arrangement"}
        if self._arrangement is not None:
            self._arrangement.write(clean)
        return {"success": True}

    def _available_layouts(self):
        metadata = getattr(self._pipeline, "pipeline_metadata", {})
        entries = []
        layouts = metadata.get("layouts", [])
        for index, filename in enumerate(layouts if isinstance(layouts, list) else []):
            if isinstance(filename, str) and filename.strip():
                entries.append((f"pipeline:{index}", "pipeline", Path(filename).stem,
                                Path(self._pipeline.path) / filename))
        if self._arrangement:
            for path in self._arrangement.custom_layouts():
                entries.append((f"custom:{path.name}", "custom", path.stem, path))
        return entries

    def restart_gui(self) -> dict:
        """Close the GUI so start() can replace the process with fresh code."""
        if self._window is None:
            return {"success": False, "error": "No GUI window is open."}
        if not self._update_lock.acquire(blocking=False):
            return {"success": False, "error": "Wait for the update to finish before restarting."}
        try:
            if self._collectra_folders:
                self._save_folder_session()
            elif self._yaml_path:
                self._save_folder_session([Path(self._yaml_path).parent])
            self._restart_requested = True
            self._window.destroy()
            return {"success": True}
        except Exception as error:
            self._restart_requested = False
            return {"success": False, "error": str(error)}
        finally:
            self._update_lock.release()

    def get_layouts(self) -> dict:
        """List workflow and named user layouts without exposing arbitrary file access."""
        metadata = getattr(self._pipeline, "pipeline_metadata", {})
        return {
            "success": True,
            "pipeline_name": metadata.get("name") or getattr(self._pipeline, "name", "Pipeline"),
            "layouts": [{"id": key, "source": source, "name": name}
                        for key, source, name, _ in self._available_layouts()],
        }

    def check_updates(self) -> dict:
        from .updates import collectra_repository, inspect_repository

        if not self._update_lock.acquire(blocking=False):
            return {"success": False, "error": "An update operation is already running."}
        try:
            targets = [
                ("pipeline", getattr(self._pipeline, "path", None), "Pipeline"),
                ("collectra", collectra_repository(), "Collectra"),
            ]
            return {"success": True, "restart_required": self._restart_required, "repositories": [
                {"id": key, **inspect_repository(root, label)} for key, root, label in targets
            ]}
        finally:
            self._update_lock.release()

    def install_update(self, target: str) -> dict:
        from .updates import collectra_repository, update_repository

        if target not in {"pipeline", "collectra"}:
            return {"success": False, "error": "Unknown update target."}
        if not self._update_lock.acquire(blocking=False):
            return {"success": False, "error": "An update operation is already running."}
        try:
            root, label = (
                (getattr(self._pipeline, "path", None), "Pipeline") if target == "pipeline"
                else (collectra_repository(), "Collectra")
            )
            repository = {"id": target, **update_repository(root, label)}
            if repository["status"] == "updated":
                self._restart_required = True
            return {"success": True, "repository": repository}
        finally:
            self._update_lock.release()

    def load_layout(self, layout_id: str) -> dict:
        from .arrangement import ArrangementStore

        for key, _, _, path in self._available_layouts():
            if key == layout_id:
                arrangement = ArrangementStore._read_file(path)
                if arrangement is not None:
                    return {"success": True, "arrangement": arrangement}
                return {"success": False, "error": "This layout is missing or invalid."}
        return {"success": False, "error": "Layout not found."}

    def reveal_layout(self, layout_id: str) -> dict:
        """Reveal a workflow or custom layout in the system file manager."""
        for key, _, _, path in self._available_layouts():
            if key == layout_id:
                return self.reveal_in_folder(str(path.resolve()))
        return {"success": False, "error": "Layout not found."}

    def save_named_layout(self, name: str, arrangement: dict) -> dict:
        from .arrangement import sanitize_arrangement

        clean = sanitize_arrangement(arrangement)
        if clean is None or self._arrangement is None:
            return {"success": False, "error": "No valid layout to save."}
        try:
            self._arrangement.save_named(name, clean)
        except (ValueError, OSError) as error:
            return {"success": False, "error": str(error)}
        return {"success": True}

    def activate_layout(self, layout_id: str) -> dict:
        """Persist a chosen layout without touching the loaded results."""
        result = self.load_layout(layout_id)
        if not result["success"] or self._arrangement is None:
            return result if not result["success"] else {"success": False, "error": "No pipeline open."}
        try:
            self._arrangement._write_file(
                self._arrangement.path, result["arrangement"], raise_errors=True
            )
        except OSError as error:
            return {"success": False, "error": str(error)}
        return {"success": True}

    def get_initial_items(self) -> dict:
        """Return startup items once the JavaScript bridge is ready."""
        return self._initial_items or {"success": True, "provided": False}

    def _load_input_paths(self, inputs) -> dict:
        """Resolve existing results without creating or rewriting any files."""
        seen = set()
        for value in inputs:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"GUI input does not exist: {path}")
            directory = path.parent if path.is_file() else path
            if (directory / "results.yaml").is_file():
                candidates = [directory]
            elif path.is_dir():
                candidates = [
                    child
                    for child in sorted(path.iterdir())
                    if child.is_dir() and child.suffix == f".{self._extension}"
                ]
            else:
                raise ValueError(
                    f"GUI input must belong to a folder containing results.yaml: {path}"
                )
            matched = False
            for candidate in candidates:
                scan = self._scan_collectra_folder(candidate)
                if scan is None:
                    continue
                matched = True
                candidate = candidate.resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                self._collectra_folders.append(
                    {
                        "name": candidate.name,
                        "path": str(candidate),
                        **scan,
                    }
                )
            if not matched:
                raise ValueError(f"No GUI results found in: {path}")
        self._parent_folder = str(Path(self._collectra_folders[0]["path"]).parent)
        self._save_folder_session()
        return {**self._folder_list_result(), "provided": True, "mode": "parent"}

    @property
    def _extension(self) -> str:
        """Use the extension owned by the live pipeline."""
        return self._pipeline.ext

    def set_window(self, window):
        """Store the window reference for use in dialogs."""
        self._window = window

    def reveal_in_folder(self, path: str) -> dict:
        """Reveal a file in the OS file manager, selected within its folder.

        Args:
            path: Absolute path to reveal (e.g. a specimen's results.yaml
                file, selected within its .grapto folder).

        Returns:
            dict with 'success' or 'error'
        """
        import subprocess

        target = Path(path)
        if not target.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        system = platform.system()
        if system == "Darwin":
            # -R reveals + selects the item in its *parent's* Finder window —
            # actual "Reveal in Finder" behavior — not the folder's own contents.
            command = ["open", "-R", str(target)]
        elif system == "Windows":
            # Explorer's reveal+select is one combined `/select,<path>` argument.
            command = ["explorer", f"/select,{target}"]
        else:
            # No universal reveal+select convention across Linux desktop
            # environments — opening the parent folder is the closest
            # approximation available via a single xdg-open call.
            command = ["xdg-open", str(target.parent)]

        try:
            subprocess.run(command)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def copy_to_clipboard(self, text: str) -> dict:
        """Copy text to the system clipboard (used by the 'Copy Path' menu item).

        The webview's own clipboard API is unavailable outside secure contexts,
        so use the operating system's clipboard tool instead.

        Returns:
            dict with 'success' or 'error'
        """
        import subprocess

        system = platform.system()
        if system == "Darwin":
            commands = [["pbcopy"]]
        elif system == "Windows":
            commands = [["clip"]]
        else:
            commands = [["xclip", "-selection", "clipboard"], ["wl-copy"]]

        for command in commands:
            try:
                subprocess.run(command, input=text, text=True, check=True)
                return {"success": True}
            except FileNotFoundError:
                continue
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {
            "success": False,
            "error": f"No clipboard tool found ({' or '.join(c[0] for c in commands)})",
        }

    def _scan_collectra_folder(self, folder_path: Path) -> dict | None:
        """
        Scan a folder for YAML and image files.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            dict with yaml_path and optional image_path, or None without results
        """
        results_path = folder_path / "results.yaml"
        image_path = None
        image_extensions = [f".{ext.lower()}" for ext in ImageFormat.__members__.keys()]

        for filename in sorted(folder_path.iterdir(), reverse=True):
            if not filename.is_file():
                continue

            if filename.suffix.lower() in image_extensions and image_path is None:
                image_path = filename

            if image_path:
                break

        if not results_path.is_file():
            return None

        return {
            "yaml_path": str(results_path),
            "image_path": str(image_path) if image_path else "",
        }

    def select_folder(self) -> dict:
        """
        Open native folder dialog and scan for YAML/image files.

        Returns:
            dict with folder_path, yaml_path, image_path (all absolute paths)
        """

        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.FOLDER
            )

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            folder_path = Path(result[0])

            scan_result = self._scan_collectra_folder(folder_path)
            if scan_result is None:
                return {
                    "success": False,
                    "error": "No results.yaml found in folder",
                }

            return {
                "success": True,
                "folder_path": str(folder_path),
                **scan_result,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def select_parent_folder(self) -> dict:
        """
        Open native folder dialog and scan for subdirectories with the specified extension.

        Returns:
            dict with parent_path and list of folder names/indices
        """
        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.FOLDER
            )

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            parent_path = Path(result[0])
            return self._scan_parent_folder(parent_path)

        except Exception as e:
            return {"success": False, "error": str(e)}

    def select_collectra_path(self) -> dict:
        """
        Open native folder dialog and load either a single collectra folder
        (if it matches the extension) or a parent folder of collectra subfolders.

        Returns:
            dict with mode "single" or "parent" and the relevant data.
        """
        if self._window is None:
            return {"success": False, "error": "Window not initialized"}

        try:
            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.FOLDER
            )

            if not result or len(result) == 0:
                return {"success": False, "error": "No folder selected"}

            selected_path = Path(result[0])

            # A previously-viewed page's yaml path is otherwise never cleared,
            # and _collectra_dirs() searches near it — so without this, asset
            # lookups (get_logo, get_theme_css) can silently keep matching a
            # file from whatever project was open before this new selection.
            self._yaml_path = None
            self._collectra_file = None
            self._graph = None

            if selected_path.suffix == f".{self._extension}":
                scan_result = self._scan_collectra_folder(selected_path)
                if scan_result is None:
                    return {
                        "success": False,
                        "error": "No YAML or image file found in folder",
                    }
                self._collectra_folders = []
                self._parent_folder = str(selected_path.parent)
                self._save_folder_session([selected_path])
                return {
                    "success": True,
                    "mode": "single",
                    "folder_path": str(selected_path),
                    "yaml_path": scan_result["yaml_path"],
                    "image_path": scan_result["image_path"],
                }

            parent_result = self._scan_parent_folder(selected_path)
            if parent_result.get("success"):
                parent_result["mode"] = "parent"
            return parent_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _scan_parent_folder(self, parent_path: Path) -> dict:
        self._parent_folder = str(parent_path)
        self._collectra_folders = []

        for entry in sorted(parent_path.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.suffix == f".{self._extension}":
                continue

            scan_result = self._scan_collectra_folder(entry)
            if scan_result is not None:
                self._collectra_folders.append(
                    {
                        "name": entry.name,
                        "path": str(entry),
                        "yaml_path": scan_result["yaml_path"],
                        "image_path": scan_result["image_path"],
                    }
                )

        self._save_folder_session()
        return self._folder_list_result()

    def _folder_list_result(self) -> dict:
        folders = [
            {
                "name": f["name"],
                "path": f["path"],
                "yaml_path": f["yaml_path"],
                "index": i,
            }
            for i, f in enumerate(self._collectra_folders)
        ]

        # Compute aggregate label statistics across all folders
        self._global_label_counts = {}
        for folder in self._collectra_folders:
            try:
                collectra_file = CollectraFile.from_data(Path(folder["path"]))
                temp_graph = CollectraGraph.from_collectra_file(collectra_file)
                label_counts = temp_graph.count_nodes_by_label(
                    type_filter="collectra.ImageCrop"
                )
                # Accumulate unique labels
                self._global_labels.update(temp_graph.get_unique_labels())
                # Accumulate counts
                for label, count in label_counts.items():
                    self._global_label_counts[label] = (
                        self._global_label_counts.get(label, 0) + count
                    )
            except Exception:
                # Skip folders that can't be loaded
                pass

        return {
            "success": True,
            "parent_path": self._parent_folder,
            "folders": folders,
            "global_label_counts": self._global_label_counts,
            "global_total": sum(self._global_label_counts.values()),
            "ext": self._extension,
        }

    def load_collectra_folder(self, index: int) -> dict:
        """
        Load a folder by its index from the previously scanned parent folder.

        Args:
            index: Index of the folder in the _collectra_folders list

        Returns:
            dict with folder_name, yaml_path, image_path
        """
        if not self._collectra_folders:
            return {
                "success": False,
                "error": "No folders loaded. Call select_parent_folder first.",
            }

        if index < 0 or index >= len(self._collectra_folders):
            return {"success": False, "error": f"Invalid folder index: {index}"}

        folder = self._collectra_folders[index]
        return {
            "success": True,
            "folder_name": folder["name"],
            "folder_path": folder["path"],
            "yaml_path": folder["yaml_path"],
            "image_path": folder["image_path"],
        }

    def get_active_node_ids(self, page_index: int) -> dict:
        """Return the labels, real types, and real result ids that belong to a page.

        `results.yaml` indexes each node by an internal id (a label plus a
        generated suffix for anything not an original source node), but the
        The pipeline DAG uses the plain label as its node id.
        Correlating against the DAG must go through `label`, not `id`.

        The frontend greys out and disables every DAG node whose id is NOT in
        active_ids, can use node_types[label] to show the node's real type,
        and node_ids[label] to show the real results.yaml id(s) — a list,
        not a single value, since a detector-type node (e.g.
        collectra.ObjectDetectionYOLO) declares one output label in the
        pipeline but can produce any number of real instances per page
        (one per detected object), all sharing that same label.
        node_parents supplies each instance's actual direct parent ids so
        lineage lists do not include unrelated instances of the same label.
        """
        if not self._collectra_folders:
            return {"success": False, "error": "No folders loaded."}
        if page_index < 0 or page_index >= len(self._collectra_folders):
            return {"success": False, "error": f"Invalid page index: {page_index}"}

        folder = self._collectra_folders[page_index]
        try:
            if (
                self._graph is not None
                and self._yaml_path is not None
                and Path(self._yaml_path).resolve()
                == Path(folder["yaml_path"]).resolve()
            ):
                graph = self._graph
            else:
                collectra_file = CollectraFile.from_data(Path(folder["path"]))
                graph = CollectraGraph.from_collectra_file(collectra_file)
            node_types: dict[str, str] = {}
            node_ids: dict[str, list[str]] = {}
            crop_labels = set()
            for node_id in graph.nodes:
                node = graph.get_node(node_id)
                if node and node.label:
                    node_types[node.label] = node.type
                    node_ids.setdefault(node.label, []).append(node_id)
                    if isinstance(node, CollectraAnnotationNode):
                        crop_labels.add(node.label)
            return {
                "success": True,
                "active_ids": list(node_types.keys()),
                "node_types": node_types,
                "node_ids": node_ids,
                "node_parents": {
                    node_id: graph.parents(node_id)
                    for node_id in graph.nodes if graph.get_node(node_id)
                },
                "crop_labels": sorted(crop_labels),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_image_base64(self, image_path: str) -> dict:
        """
        Read an image file and return its base64 data.

        Args:
            image_path: Absolute path to the image file

        Returns:
            dict with base64-encoded data URI
        """
        try:
            extension = Path(image_path).suffix.lower()
            mime_type = ImageFormat[extension[1:].upper()].value

            with open(image_path, "rb") as f:
                image_data = f.read()

            base64_data = base64.b64encode(image_data).decode("utf-8")

            return {"success": True, "data": f"data:{mime_type};base64,{base64_data}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _display_context(self, rgb_view: int = 0):
        from collectra.display import DisplayContext

        if self._graph is None or self._yaml_path is None:
            raise ValueError("No page loaded")
        records = {
            node_id: {
                **self._graph.get_node(node_id).model_dump(),
                "label": self._graph.get_node(node_id).label,
            }
            for node_id in self._graph.nodes
            if self._graph.get_node(node_id)
        }
        return DisplayContext(records, Path(self._yaml_path).parent, rgb_view=rgb_view, max_pixels=self._max_pixels)

    def get_artefact_display(self, node_id: str = "", rgb_view: int = 0) -> dict:
        """Ask the actual artefact class how it should be presented."""
        try:
            context = self._display_context(rgb_view=rgb_view)
            if not node_id:
                from collectra.types.images import Image, ImageCrop
                from collectra.utils import load_class_from_string

                for candidate in context.records:
                    try:
                        cls = load_class_from_string(context.records[candidate]["type"])
                    except (ImportError, AttributeError):
                        continue
                    if issubclass(cls, Image) and not issubclass(cls, ImageCrop):
                        node_id = candidate
                        break
            node_id = self._graph.resolve_id(node_id)
            item = context.artefact(node_id)
            view = item.display(context)
            return {
                "success": True,
                "view": view,
                "id": node_id,
                "type": context.records[node_id]["type"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rotate_image_clockwise(self, node_id: str) -> dict:
        """Persist a clockwise quarter-turn for an image and descendant images."""
        from collectra.types.images import Image, Orientation
        from collectra.types.links import Link
        from collectra.utils import load_class_from_string

        changes = []
        try:
            context = self._display_context()
            item = context.artefact(self._graph.resolve_id(node_id))
            if isinstance(item, Link):
                item = item.resolve()
            if not isinstance(item, Image):
                raise ValueError("Only image artefacts can be rotated")
            pending, seen = [item.id], set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(self._graph.children(current))
                node = self._graph.get_node(current)
                if node and issubclass(load_class_from_string(node.type), Image):
                    orientation = Orientation.from_string(node.orientation)
                    next_orientation = Orientation(
                        (orientation.value + 1) % 4
                    ).to_string()
                    changes.append((node, node.orientation, next_orientation))
            for node, previous, updated in changes:
                node.orientation = updated
            self._save_collectra_file()
            return {"success": True, "updated_ids": [node.id for node, _, _ in changes]}
        except Exception as error:
            for node, previous, updated in changes:
                node.orientation = previous
            return {"success": False, "error": str(error)}

    def update_segmentation_mask(
        self, node_id: str, strokes: list, results_path: str, orientation: str
    ) -> dict:
        """Save strokes to their original results page, even after navigation."""
        from collectra.display import DisplayContext
        from collectra.displays.image_segmentation import segmentation
        from collectra.gui.segmentation import paint_mask
        from collectra.types.links import Link

        try:
            with self._text_save_lock:
                target = Path(results_path).resolve()
                if target.name != "results.yaml" or not target.is_file():
                    raise ValueError("Mask edits must target a results.yaml file")
                collectra_file = CollectraFile.from_data(target.parent)
                graph = CollectraGraph.from_collectra_file(collectra_file)
                records = {
                    key: {**graph.get_node(key).model_dump(), "label": graph.get_node(key).label}
                    for key in graph.nodes if graph.get_node(key)
                }
                context = DisplayContext(records, target.parent, max_pixels=self._max_pixels)
                item = context.artefact(graph.resolve_id(node_id))
                if isinstance(item, Link):
                    item = item.resolve()
                changed = paint_mask(item, strokes, orientation)
                return {"success": True, "changed": changed, "segmentation": segmentation(context, item)}
        except Exception as error:
            return {"success": False, "error": str(error)}

    def get_display_annotations(self, node_id: str) -> dict:
        try:
            context = self._display_context()
            image = context.artefact(self._graph.resolve_id(node_id))
            from collectra.types.links import Link

            if type(image) is Link:
                image = image.resolve()
            return {"success": True, "rows": context.annotations(image)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_display_text(
        self, node_id: str, content: str, results_path: str | None = None
    ) -> dict:
        """Save changed text to its captured page, even after navigation."""
        try:
            from collectra.display import DisplayContext

            with self._text_save_lock:
                target = Path(results_path or self._yaml_path).resolve()
                if target.name != "results.yaml":
                    raise ValueError("Text edits must target a results.yaml file")
                if self._yaml_path and target == Path(self._yaml_path).resolve():
                    graph, collectra_file = self._graph, self._collectra_file
                else:
                    collectra_file = CollectraFile.from_data(target.parent)
                    graph = CollectraGraph.from_collectra_file(collectra_file)
                records = {
                    key: {**graph.get_node(key).model_dump(), "label": graph.get_node(key).label}
                    for key in graph.nodes if graph.get_node(key)
                }
                context = DisplayContext(records, target.parent, max_pixels=self._max_pixels)
                item = context.artefact(graph.resolve_id(node_id))
                view = item.display(context)
                if view.get("kind") != "text" or not view.get("editable"):
                    raise ValueError("This artefact does not expose editable text")
                target_id = view["target_id"]
                path = context.text_path(target_id)
                if path:
                    if path.read_text(encoding="utf-8") == content:
                        return {"success": True, "changed": False}
                    path.write_text(content, encoding="utf-8")
                else:
                    if graph.get_data(target_id) == content:
                        return {"success": True, "changed": False}
                    previous = graph.get_data(target_id)
                    graph.set_data(target_id, content)
                    try:
                        collectra_file.data = graph.to_data()
                        collectra_file.save()
                    except Exception:
                        graph.set_data(target_id, previous)
                        collectra_file.data = graph.to_data()
                        raise
                return {"success": True, "changed": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_yaml(self, path: str) -> dict:
        with self._text_save_lock:
            return self._load_yaml(path)

    def _load_yaml(self, path: str) -> dict:
        """
        Load a YAML file and build the annotation graph.

        Args:
            path: Path to the YAML file

        Returns:
            dict with 'success', 'node_count', 'edge_count', or 'error'
        """
        try:
            collectra_path = Path(path)
            directory = (
                collectra_path if collectra_path.is_dir() else collectra_path.parent
            )
            collectra_file = CollectraFile.from_data(directory)
            graph = CollectraGraph.from_collectra_file(collectra_file)
            self._collectra_file = collectra_file
            self._graph = graph
            previous_path = self._yaml_path
            self._yaml_path = str(self._collectra_file.results_path)
            if self._session_cache is not None and previous_path != self._yaml_path:
                self._session_cache.write_active(
                    str(self._collectra_file.results_path.parent.resolve())
                )
            if not self._collectra_folders:
                self._save_folder_session([self._collectra_file.results_path.parent])

            # Accumulate labels from this graph
            self._global_labels.update(self._graph.get_unique_labels())

            return {"success": True, "path": self._yaml_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_display_value(self, node_id: str) -> dict:
        """
        Compute display value for an annotation.

        Args:
            node_id: The annotation ID

        Returns:
            dict with 'value', 'source_id', 'reason', or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            result = self._graph.compute_display_value(self._graph.resolve_id(node_id))
            return {
                "success": True,
                "value": result.value,
                "source_id": result.source_id,
                "crop_region": result.crop_region,
                "reason": result.reason,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_texts(self) -> dict:
        """Return markdown and TEI content for the current page folder."""
        if self._yaml_path is None:
            return {"success": False, "error": "No page loaded."}
        page_dir = Path(self._yaml_path).parent
        result: dict = {"success": True}
        for key, filename in (("markdown", "markdown.md"), ("tei", "tei.xml")):
            p = page_dir / filename
            if p.exists():
                result[key] = p.read_text(encoding="utf-8")
                result[f"{key}_found"] = True
            else:
                result[key] = None
                result[f"{key}_found"] = False
        return result

    def update_page_text(self, kind: str, content: str) -> dict:
        """Write markdown or TEI content back to the current page's folder.

        Args:
            kind: "markdown" or "tei"
            content: the full text to write

        Returns:
            dict with 'success' or 'error'
        """
        if self._yaml_path is None:
            return {"success": False, "error": "No page loaded."}

        filenames = {"markdown": "markdown.md", "tei": "tei.xml"}
        filename = filenames.get(kind)
        if filename is None:
            return {"success": False, "error": f"Unknown kind: {kind}"}

        try:
            page_dir = Path(self._yaml_path).parent
            (page_dir / filename).write_text(content, encoding="utf-8")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_all_nodes(self) -> dict:
        """
        Get all node IDs in the graph.

        Returns:
            dict with 'nodes' list or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        return {"success": True, "nodes": self._graph.nodes}

    def get_node_info(self, node_id: str) -> dict:
        """
        Get detailed information about a node.

        Args:
            node_id: The annotation ID

        Returns:
            dict with node type, data, children, parents, or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        node_id = self._graph.resolve_id(node_id)
        if node_id not in self._graph.nodes:
            return {"success": False, "error": f"Node '{node_id}' not found"}

        return {
            "success": True,
            "id": node_id,
            "type": self._graph.get_type(node_id),
            "data": self._graph.get_data(node_id),
            "children": self._graph.children(node_id),
            "parents": self._graph.parents(node_id),
        }

    def get_nodes_by_type(self, type_filter: str) -> dict:
        """
        Get all nodes containing a type substring.

        Args:
            type_filter: Substring to match in node types (e.g., "ImageCrop", "Text")

        Returns:
            dict with 'nodes' list of matching node IDs
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        matching = [
            node_id
            for node_id in self._graph.nodes
            if type_filter in self._graph.get_type(node_id)
        ]

        return {"success": True, "nodes": matching, "count": len(matching)}

    def get_available_labels(self) -> dict:
        """
        Get all unique labels accumulated across all loaded files.

        Returns:
            dict with 'success' and 'labels' list (sorted)
        """
        try:
            labels = sorted(self._global_labels)
            return {"success": True, "labels": labels}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_label_statistics(self) -> dict:
        """
        Return counts of nodes per label in current graph.

        Returns:
            dict with 'success', 'label_counts', and 'total'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded"}

        # Count only ImageCrop types (annotations visible in grid)
        label_counts = self._graph.count_nodes_by_label(
            type_filter="collectra.ImageCrop"
        )

        return {
            "success": True,
            "label_counts": label_counts,
            "total": sum(label_counts.values()),
        }

    def get_global_label_statistics(self) -> dict:
        """
        Return aggregated label counts across all files in parent folder.

        Returns:
            dict with 'success', 'label_counts', and 'total'
        """
        return {
            "success": True,
            "label_counts": self._global_label_counts,
            "total": sum(self._global_label_counts.values()),
        }

    def get_all_nodes_for_grid(self) -> dict:
        """
        Return all nodes formatted for AG Grid rows.

        Returns:
            dict with 'rows' list containing node data for grid display
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        rows = []
        for node_id in self._graph.nodes:
            node_display_value: NodeDisplayValue = self._graph.compute_display_value(
                node_id
            )
            rows.append(
                {
                    "id": node_id,
                    "type": self._graph.get_type(node_id),
                    "data": str(self._graph.get_data(node_id)),
                    "name": self._graph.get_name(node_id),
                    "label": self._graph.get_label(node_id),
                    "displayValue": node_display_value.value,
                    "crop_region": node_display_value.crop_region,
                    "displaySourceId": node_display_value.source_id,
                    "reason": node_display_value.reason,
                    "parents": ", ".join(self._graph.parents(node_id)),
                    "children": ", ".join(self._graph.children(node_id)),
                    "locked": node_display_value.locked,
                }
            )

        return {"success": True, "rows": rows}

    def update_node_data(
        self, node_id: str, new_data: str, crop_id: str | None = None
    ) -> dict:
        """
        Update the data of a specific node.

        Args:
            node_id: The annotation ID
            new_data: New data to set for the node

        Returns:
            dict with 'success' or 'error'
        """
        try:
            if self._graph is None:
                raise ValueError("No graph loaded. Call load_yaml first.")
            # node_id may be "" when creating a new Text node via crop_id — only
            # resolve a real, non-empty id (an empty label would never match).
            resolved_id = self._graph.resolve_id(node_id) if node_id else node_id
            self._graph.set_data(resolved_id, new_data, crop_id)
            self._save_collectra_file()
            return self.get_all_nodes_for_grid()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_node_coordinates(
        self, node_id: str, crop_region: dict, view_id: str = ""
    ) -> dict:
        """
        Update the crop region coordinates of a specific node.

        Args:
            node_id: The annotation ID
            crop_region: dict with x_center, y_center, width_relative, height_relative

        Returns:
            dict with 'success' and updated rows, or 'error'
        """
        try:
            if self._graph is None:
                raise ValueError("No graph loaded. Call load_yaml first.")
            if view_id:
                context = self._display_context()
                crop_region = context.region(
                    context.artefact(view_id), crop_region, inverse=True
                )
            self._graph.set_crop_region(self._graph.resolve_id(node_id), crop_region)
            self._save_collectra_file()
            return self.get_all_nodes_for_grid()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _next_annotation_id(self, label: str) -> str:
        """Matches the pipeline's scheme (label+N) — reuses the smallest
        suffix a deletion freed up, rather than always growing past the
        highest one ever used. Shared with preview_annotation_id."""
        existing_numbers = set()
        for node_id in self._graph.nodes:
            node = self._graph.get_node(node_id)
            if node and node.label == label:
                suffix = node_id[len(label) :]
                if suffix.isdigit():
                    existing_numbers.add(int(suffix))
        n = 1
        while n in existing_numbers:
            n += 1
        return f"{label}{n}"

    def preview_annotation_id(self, label: str) -> dict:
        """What create_annotation would name the next box with this label, without creating anything."""
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}
        return {"success": True, "id": self._next_annotation_id(label)}

    def create_annotation(
        self,
        crop_region: dict,
        label: str,
        parent_id: str,
        image_path: str = "",
        name: str = "",
        crop_id: str = "",
        view_id: str = "",
    ) -> dict:
        """
        Create a new ImageCrop annotation with a Text child.

        Args:
            crop_region: dict with x_center, y_center, width_relative, height_relative
            name: optional per-instance display name, distinct from label
            crop_id: explicit node id; auto-numbered (label + N) when blank

        Returns:
            dict with updated grid data or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        if not image_path:
            return {
                "success": False,
                "error": "Image path is required to create annotation.",
            }

        try:
            if view_id:
                context = self._display_context()
                image = context.artefact(view_id)
                crop_region = context.region(image, crop_region, inverse=True)
                image_path = str(image.get_path())
                parent_id = parent_id or image.id
            # Find root Image node
            root_image_id = parent_id if view_id else None
            for node_id in self._graph.nodes:
                node_type = self._graph.get_type(node_id)
                if "Image" in node_type and "ImageCrop" not in node_type:
                    root_image_id = node_id
                    break

            if root_image_id is None:
                return {"success": False, "error": "No root Image node found in graph."}

            orientation = "north"

            if parent_id:
                parent_node = self._graph.get_node(parent_id)
                if parent_node:
                    orientation = parent_node.orientation

            crop_id = crop_id.strip() or self._next_annotation_id(label)
            if crop_id in self._graph.nodes:
                return {"success": False, "error": f"Id '{crop_id}' already exists."}

            if self._yaml_path is None:
                raise ValueError("YAML path is not set.")

            asset_path = Path(image_path)
            if view_id:
                try:
                    asset_path = asset_path.relative_to(context.directory)
                except ValueError:
                    pass
            else:
                asset_path = Path(asset_path.name)
            # Create ImageCrop node
            crop_data = {
                "label": label,
                "type": "collectra.ImageCrop",
                "id": crop_id,
                "parents": parent_id if parent_id else root_image_id,
                "orientation": orientation,
                "data": str(asset_path),
                "name": name,
                **crop_region,
            }
            self._graph.add_node(crop_data)
            self._global_labels.add(label)
            # Update global label counts if in parent folder mode
            if self._collectra_folders:
                self._global_label_counts[label] = (
                    self._global_label_counts.get(label, 0) + 1
                )
            # Save and return updated grid
            self._save_collectra_file()
            result = self.get_all_nodes_for_grid()
            # Include updated global stats in response
            if self._collectra_folders:
                result["global_label_counts"] = self._global_label_counts
                result["global_total"] = sum(self._global_label_counts.values())
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _node_and_descendants(self, node_id: str) -> list[str]:
        """The node followed by every descendant (any type, any depth),
        breadth first, each id once."""
        ids = []
        queue = [node_id]
        seen = {node_id}
        while queue:
            current = queue.pop(0)
            ids.append(current)
            for child_id in self._graph.children(current):
                if child_id not in seen:
                    seen.add(child_id)
                    queue.append(child_id)
        return ids

    def search_artefacts(self, query: str) -> dict:
        """
        Find the artefacts on the open page that contain ``query``
        (case-insensitive) in their content: the text of a Text artefact or
        a custom name. DAG labels and ids are not searched.

        Returns:
            dict with 'matches': one entry per artefact, in page order, each
            with id, DAG label (to place it in the tabs), type, the first
            field that matched (text, name) and a snippet around the match
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        needle = (query or "").strip().lower()
        if not needle:
            return {"success": True, "matches": []}

        from collectra.types.texts import Text
        from collectra.utils import load_class_from_string

        context = self._display_context()
        matches = []
        for node_id in self._nodes_top_down():
            node = self._graph.get_node(node_id)
            if node is None:
                continue
            fields = []
            try:
                if issubclass(load_class_from_string(node.type), Text):
                    path = context.text_path(node_id)
                    if path:
                        text = path.read_text(encoding="utf-8")
                    elif context.records.get(node_id, {}).get("path"):
                        text = (
                            ""  # its file is missing: nothing to search, not its path
                        )
                    else:
                        text = node.data
                    fields.append(("text", str(text or "")))
            except Exception:
                # One artefact that cannot be read (unknown type, a type that is not
                # a class, a file that is not text) must not stop the whole search.
                pass
            fields.append(("name", node.name or ""))
            for field, value in fields:
                found = re.search(re.escape(needle), value, re.IGNORECASE)
                position = found.start() if found else -1
                if position != -1:
                    matches.append(
                        {
                            "id": node_id,
                            "label": node.label,
                            "type": node.type,
                            "field": field,
                            "snippet": self._snippet(
                                value, position, len(found.group())
                            ),
                        }
                    )
                    break
        # Walk the matches the way the pipeline reads; labels it does not know
        # (made by hand) come last. The sort is stable, so within one label the
        # order of the page is kept.
        ranks = self._pipeline_label_ranks()
        matches.sort(key=lambda match: ranks.get(match["label"], len(ranks)))
        return {"success": True, "matches": matches}

    def _text_artefact_labels(self) -> list[str]:
        """The pipeline's text artefact labels, in the order the pipeline reads."""
        from collectra.types.base import ArtefactNode
        from collectra.types.texts import Text

        try:
            self._pipeline._ensure_workflow_connected()  # builds the nodes
            manager = self._pipeline.node_manager
            labels = []
            for name in manager.get_node_names():
                node = manager.resolve_node(name)
                if isinstance(node, ArtefactNode) and any(
                    isinstance(type_, type) and issubclass(type_, Text)
                    for type_ in node.types
                ):
                    labels.append(name)
        except Exception:
            return []
        ranks = self._pipeline_label_ranks()
        return sorted(labels, key=lambda label: ranks.get(label, len(ranks)))

    def _page_text_nodes(self, graph) -> dict[str, list[str]]:
        """The ids of a page's text artefacts, by DAG label."""
        from collectra.types.texts import Text
        from collectra.utils import load_class_from_string

        nodes: dict[str, list[str]] = {}
        for node_id in graph.nodes:
            node = graph.get_node(node_id)
            if node is None:
                continue
            try:
                if issubclass(load_class_from_string(node.type), Text):
                    nodes.setdefault(node.label, []).append(node_id)
            except Exception:
                continue  # an unknown type is not a text
        return nodes

    def _page_text_cells(
        self, graph, directory: Path, limit: int
    ) -> tuple[dict[str, str], list[str]]:
        """The text of each text artefact on a page, by DAG label, and the
        labels that can be edited from the table: those with exactly one
        artefact, shown in full. A label with several artefacts has them
        joined with " | "; each is cut at ``limit``."""
        from collectra.display import DisplayContext

        records = {
            node_id: {**graph.get_node(node_id).model_dump(), "label": graph.get_node(node_id).label}
            for node_id in graph.nodes
            if graph.get_node(node_id)
        }
        context = DisplayContext(records, directory, max_pixels=self._max_pixels)
        cells: dict[str, str] = {}
        editable = []
        for label, node_ids in self._page_text_nodes(graph).items():
            values, whole = [], True
            for node_id in node_ids:
                record = records[node_id]
                try:
                    path = context.text_path(node_id)
                    if path:
                        text = path.read_text(encoding="utf-8")
                    elif record.get("path"):
                        text = ""  # its file is missing: show nothing, not its path
                    else:
                        text = record.get("data") or ""
                except Exception:
                    whole = False  # one unreadable artefact must not blank the table
                    continue
                text = str(text)
                if len(text) > limit:
                    text = text[:limit] + "\u2026"
                    whole = False
                values.append(text)
            if not values:
                continue
            cells[label] = " | ".join(values)
            if whole and len(node_ids) == 1:
                editable.append(label)
        return cells, editable

    def get_table_data(self, max_chars: int = 2000) -> dict:
        """
        Return the text artefacts of every loaded page for the table view.

        Returns:
            dict with 'columns' (the text artefact labels, in pipeline order)
            and 'rows': one per loaded folder, in sidebar order, each with
            'index', 'name', the text of each label that page has, and
            '__editable': the labels whose text set_table_cells can change.
        """
        if not self._collectra_folders:
            return {"success": True, "columns": [], "rows": []}
        columns = self._text_artefact_labels()
        rows = []
        for index, folder in enumerate(self._collectra_folders):
            row = {"index": index, "name": folder["name"]}
            try:
                if (
                    self._graph is not None
                    and self._yaml_path is not None
                    and Path(self._yaml_path).resolve()
                    == Path(folder["yaml_path"]).resolve()
                ):
                    graph = self._graph
                else:
                    graph = CollectraGraph.from_collectra_file(
                        CollectraFile.from_data(Path(folder["path"]))
                    )
                cells, editable = self._page_text_cells(
                    graph, Path(folder["path"]), max_chars
                )
            except Exception:
                cells, editable = {}, []  # a folder that cannot be loaded is still a row
            for label, text in cells.items():
                if label not in columns:
                    columns.append(label)  # made by hand, not in the pipeline
                row[label] = text
            row["__editable"] = editable
            rows.append(row)
        return {"success": True, "columns": columns, "rows": rows}

    def export_table_csv(self, headers: dict | None = None) -> dict:
        """
        Ask where to save, then write the table as a CSV: a "File" column and
        one column per text artefact, one row per loaded page, with every text
        in full.

        Args:
            headers: optional column titles by text artefact label, as the
                table shows them

        Returns:
            dict with 'success' and 'path', or 'cancelled' when no file was
            chosen
        """
        if self._window is None:
            return {"success": False, "error": "Window not initialized"}
        try:
            table = self.get_table_data(max_chars=10**12)
            if not table.get("columns") and not table.get("rows"):
                return {"success": False, "error": "No files are loaded"}
            result = self._window.create_file_dialog(
                dialog_type=webview.FileDialog.SAVE,
                save_filename="table.csv",
                file_types=("CSV files (*.csv)",),
            )
            if not result:
                return {"success": False, "cancelled": True}
            path = Path(result if isinstance(result, (str, Path)) else result[0])
            if not path.suffix:
                path = path.with_suffix(".csv")
            columns = table["columns"]
            headers = headers or {}
            # utf-8-sig so that Excel reads non-ASCII text correctly
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["File"] + [headers.get(c) or c for c in columns])
                for row in table["rows"]:
                    writer.writerow([row["name"]] + [row.get(c, "") for c in columns])
            return {"success": True, "path": str(path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_table_cells(self, edits: list[dict]) -> dict:
        """
        Save text pasted or typed into the table.

        Args:
            edits: each {'index': the page's index in the sidebar, 'label': the
                text artefact's DAG label, 'text': the new text}

        Only a label with exactly one text artefact on its page can be set;
        the rest are skipped, not created. Every other edit is still saved
        when one fails.

        Returns:
            dict with 'saved' (the edits that were written, or already held
            that text) and 'skipped' (each edit that was not, with a 'reason')
        """
        saved, skipped = [], []
        by_page: dict[int, list[dict]] = {}
        for edit in edits or []:
            by_page.setdefault(edit.get("index"), []).append(edit)
        for index, page_edits in by_page.items():
            def skip(edit, reason):
                skipped.append({"index": index, "label": edit.get("label"), "reason": reason})

            if not isinstance(index, int) or not 0 <= index < len(self._collectra_folders):
                for edit in page_edits:
                    skip(edit, "No such page")
                continue
            folder = self._collectra_folders[index]
            try:
                if (
                    self._graph is not None
                    and self._yaml_path is not None
                    and Path(self._yaml_path).resolve()
                    == Path(folder["yaml_path"]).resolve()
                ):
                    graph = self._graph
                else:
                    graph = CollectraGraph.from_collectra_file(
                        CollectraFile.from_data(Path(folder["path"]))
                    )
                nodes = self._page_text_nodes(graph)
            except Exception as e:
                for edit in page_edits:
                    skip(edit, str(e))
                continue
            for edit in page_edits:
                node_ids = nodes.get(edit.get("label"), [])
                if len(node_ids) != 1:
                    skip(edit, "Needs exactly one text artefact on the page")
                    continue
                result = self.update_display_text(
                    node_ids[0], str(edit.get("text", "")), folder["yaml_path"]
                )
                if result.get("success"):
                    saved.append({"index": index, "label": edit["label"]})
                else:
                    skip(edit, result.get("error", "Could not save"))
        return {"success": True, "saved": saved, "skipped": skipped}

    def _pipeline_label_ranks(self) -> dict[str, int]:
        from .pipeline_order import pipeline_label_ranks

        try:
            return pipeline_label_ranks(self._pipeline.gui_graph())
        except Exception:
            return {}

    def _nodes_top_down(self) -> list[str]:
        """Every node, each followed by its descendants (the image, then its
        crops, then their texts), so results read in the order of the page."""
        ordered, seen = [], set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            ordered.append(node_id)
            for child_id in self._graph.children(node_id):
                visit(child_id)

        for node_id in self._graph.nodes:
            if not self._graph.parents(node_id):
                visit(node_id)
        for node_id in self._graph.nodes:  # anything only reachable around a cycle
            visit(node_id)
        return ordered

    @staticmethod
    def _snippet(value: str, position: int, length: int, context: int = 40) -> str:
        """The match with a little text either side, trimmed with ellipses."""
        start = max(0, position - context)
        end = min(len(value), position + length + context)
        snippet = " ".join(value[start:end].split())
        return ("…" if start > 0 else "") + snippet + ("…" if end < len(value) else "")

    def get_delete_preview(self, node_id: str) -> dict:
        """
        List what delete_annotation would remove along with a node, without
        changing anything.

        Args:
            node_id: The annotation ID (or DAG label) about to be deleted

        Returns:
            dict with the resolved 'id' and 'children' (each with id, label
            and type, nearest first), or 'error' on failure
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        node_id = self._graph.resolve_id(node_id)
        if node_id not in self._graph.nodes:
            return {"success": False, "error": f"Node '{node_id}' not found"}

        children = []
        seen = {node_id}

        def walk(current: str, depth: int) -> None:
            for child_id in self._graph.children(current):
                if child_id in seen:
                    continue
                seen.add(child_id)
                node = self._graph.get_node(child_id)
                children.append(
                    {
                        "id": child_id,
                        "label": node.label if node else "",
                        "type": node.type if node else "",
                        "depth": depth,
                    }
                )
                walk(child_id, depth + 1)

        walk(node_id, 1)
        return {
            "success": True,
            "id": node_id,
            "children": children,
            "keeps_self": not self._graph.parents(node_id),
        }

    def _remove_nodes(self, to_delete: list[str]) -> dict:
        """Remove the given nodes, update label counts, save and return the
        refreshed grid plus the removed ids."""
        deleted_labels = [
            node.label
            for node_id in to_delete
            if (node := self._graph.get_node(node_id))
        ]

        # Deepest-reached first, so a node's own children are already gone
        # before it is.
        for node_id in reversed(to_delete):
            self._graph.remove_node(node_id)

        # Update global label counts if in parent folder mode
        if self._collectra_folders:
            for label in deleted_labels:
                if label in self._global_label_counts:
                    self._global_label_counts[label] -= 1
                    if self._global_label_counts[label] <= 0:
                        del self._global_label_counts[label]

        self._save_collectra_file()
        result = self.get_all_nodes_for_grid()
        # Every id actually removed — the frontend needs this to close any
        # tab left open on a cascaded descendant.
        result["deleted_ids"] = to_delete
        if self._collectra_folders:
            result["global_label_counts"] = self._global_label_counts
            result["global_total"] = sum(self._global_label_counts.values())
        return result

    def delete_nested(self, node_id: str) -> dict:
        """
        Delete everything nested under a node but keep the node itself (used
        for the page image, which cannot be removed).

        Returns:
            dict with updated grid data and 'deleted_ids', or 'error'
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            node_id = self._graph.resolve_id(node_id)
            if node_id not in self._graph.nodes:
                return {"success": False, "error": f"Node '{node_id}' not found"}
            return self._remove_nodes(self._node_and_descendants(node_id)[1:])
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_annotation(self, node_id: str) -> dict:
        """
        Delete an ImageCrop annotation and everything nested under it —
        cascades to its Text children, any child ImageCrops (sub-crops), and
        so on at any depth. Deleting a crop that still has children used to
        leave them behind pointing at a now-nonexistent parent id (a dangling
        reference the graph library then treats as a bare placeholder node,
        breaking id-uniqueness checks for that id ever after) — cascading is
        the only way to guarantee no orphans.

        Args:
            node_id: The ImageCrop annotation ID to delete

        Returns:
            dict with updated grid data on success, or 'error' on failure
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            node_id = self._graph.resolve_id(node_id)

            # Collect every descendant before deleting anything — walking the
            # graph while removing nodes from it would skip some.
            return self._remove_nodes(self._node_and_descendants(node_id))

        except Exception as e:
            return {"success": False, "error": str(e)}

    def rename_annotation(
        self,
        node_id: str,
        name: str,
        label: str | None = None,
        new_id: str | None = None,
    ) -> dict:
        """
        Sets an ImageCrop annotation's display name. label and new_id are
        optional; omitted, that field is left as-is.

        Args:
            node_id: The ImageCrop annotation ID to rename
            name: The new name
            label: optional new label
            new_id: optional new id — re-points every parent reference to match

        Returns:
            dict with updated grid data on success, or 'error' on failure
        """
        if self._graph is None:
            return {"success": False, "error": "No graph loaded. Call load_yaml first."}

        try:
            node_id = self._graph.resolve_id(node_id)
            if new_id and new_id != node_id:
                self._graph.rename_id(node_id, new_id)
                node_id = new_id
            if label is not None:
                self._graph.set_label(node_id, label)
            self._graph.set_name(node_id, name)
            self._save_collectra_file()
            return self.get_all_nodes_for_grid()

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _save_collectra_file(self) -> None:
        """Persist GUI edits through the shared Collectra file writer."""
        if self._collectra_file is None or self._graph is None:
            raise ValueError("No Collectra file loaded to save to.")
        self._collectra_file.data = self._graph.to_data()
        self._collectra_file.save()

    def _collectra_dirs(self) -> list[Path]:
        """Return all .collectra directories to search, in priority order.

        Search order: pipeline path → root itself → root's children → root's parent's children.
        The parent search lets the user open a book sub-folder (e.g. Livy-Books21-25/)
        while the .collectra lives as a sibling inside the project root (LivyProject/).
        """
        dirs: list[Path] = []
        pipeline_path = self._pipeline.path
        if pipeline_path.is_dir():
            dirs.append(pipeline_path)
        roots: list[Path] = []
        if self._parent_folder:
            roots.append(Path(self._parent_folder))
        if self._yaml_path:
            roots.append(Path(self._yaml_path).parent.parent)
        seen = set()

        def _scan(directory: Path) -> None:
            """Add any .collectra dirs found directly inside `directory`."""
            if not directory.is_dir():
                return
            if directory.suffix == ".collectra" and directory not in seen:
                dirs.append(directory)
                seen.add(directory)
                return
            try:
                for entry in directory.iterdir():
                    if (
                        entry.is_dir()
                        and entry.suffix == ".collectra"
                        and entry not in seen
                    ):
                        dirs.append(entry)
                        seen.add(entry)
            except (PermissionError, OSError):
                pass

        for root in roots:
            _scan(root)  # look inside the selected folder
            _scan(root.parent)  # look inside the parent (sibling .collectra case)

        return dirs

    def _pipeline_metadata(self) -> dict:
        """Return metadata retained by the live pipeline object."""
        return self._pipeline.pipeline_metadata.copy()

    def get_pipeline_graph(self) -> dict:
        """Return the graph already constructed by the live pipeline."""
        return {"success": True, **self._pipeline.gui_graph()}

    def get_theme_css(self) -> dict:
        """Load the CSS theme named in pipeline metadata, anywhere in the .collectra tree.

        Falls back to a file literally named theme.css, then any CSS file, when no
        named file is found.
        """
        named = self._pipeline_metadata().get("theme", "")
        for directory in self._collectra_dirs():
            # 1. Exact name from metadata
            if named:
                try:
                    hits = list(directory.rglob(named))
                except (PermissionError, OSError):
                    hits = []
                if hits and hits[0].suffix.lower() == ".scss":
                    try:
                        return {
                            "success": True,
                            "css": compile_theme(hits[0]),
                            "path": str(hits[0]),
                        }
                    except Exception as e:
                        # A broken theme must not leave the app unstyled
                        return {**self._default_theme_css(), "warning": str(e)}
                if hits:
                    try:
                        with open(hits[0], "r") as f:
                            return {
                                "success": True,
                                "css": f.read(),
                                "path": str(hits[0]),
                            }
                    except Exception as e:
                        return {"success": False, "error": str(e)}

            # 2. Fallback: prefer a file named theme.css, else any CSS file
            try:
                css_files = [
                    f for f in directory.iterdir() if f.suffix == ".css" and f.is_file()
                ]
            except (PermissionError, OSError):
                continue
            preferred = next((f for f in css_files if f.stem == "theme"), None)
            target = preferred or (css_files[0] if css_files else None)
            if target:
                try:
                    with open(target, "r") as f:
                        return {"success": True, "css": f.read(), "path": str(target)}
                except Exception as e:
                    return {"success": False, "error": str(e)}

        return self._default_theme_css()

    def _default_theme_css(self) -> dict:
        """The CSS of the bundled default theme."""
        asset_path = Path(get_resource_path("assets/theme.css"))
        if asset_path.is_file():
            try:
                with open(asset_path, "r") as f:
                    return {"success": True, "css": f.read(), "path": str(asset_path)}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "No CSS theme file found"}

    def _encode_logo(self, path: Path) -> dict:
        """Encode a logo file (PNG/SVG/etc.) as a base64 data URI with the right MIME type."""
        try:
            ext = path.suffix.lower()
            if ext == ".svg":
                mime = "image/svg+xml"
            else:
                mime = ImageFormat[ext[1:].upper()].value
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return {"success": True, "data": f"data:{mime};base64,{data}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_logo(self) -> dict:
        """Find the logo named in pipeline metadata (PNG or SVG), anywhere in the .collectra tree.

        Falls back to globbing for a logo-ish image when no named file is found,
        then to Collectra's own bundled logo when the project has none at all —
        callers rely on this always succeeding so a project with no logo doesn't
        keep showing whichever project's logo loaded previously.
        """
        named = self._pipeline_metadata().get("logo", "")
        image_exts = {f".{e.lower()}" for e in ImageFormat.__members__} | {".svg"}
        for directory in self._collectra_dirs():
            # 1. Exact name from metadata
            if named:
                try:
                    hits = list(directory.rglob(named))
                except (PermissionError, OSError):
                    hits = []
                if hits:
                    return self._encode_logo(hits[0])
            # 2. Glob fallback: prefer a "logo"-named image, else any image/SVG
            try:
                candidates = [
                    f for f in directory.rglob("*") if f.suffix.lower() in image_exts
                ]
            except (PermissionError, OSError):
                continue
            logos = [f for f in candidates if "logo" in f.stem.lower()]
            target = logos[0] if logos else (candidates[0] if candidates else None)
            if target:
                return self._encode_logo(target)

        # Fall back to bundled default asset
        asset_path = Path(get_resource_path("assets/CollectraLogoAndTitle.svg"))
        if asset_path.is_file():
            return self._encode_logo(asset_path)

        return {"success": False, "error": "No logo file found"}


def _set_macos_app_name(name: str) -> None:
    """Set the app identity before Cocoa creates its native application menu."""
    if platform.system() != "Darwin":
        return
    try:
        from Foundation import NSBundle, NSProcessInfo

        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = name
        info["CFBundleDisplayName"] = name
        NSProcessInfo.processInfo().setProcessName_(name)
    except Exception:
        # A packaged launcher also supplies these names in its Info.plist.
        import logging

        logging.getLogger(__name__).warning("Could not set macOS application name", exc_info=True)


def _disable_macos_tabbing() -> None:
    """Disable macOS Window Tab Bar for the pywebview window."""
    import platform

    if platform.system() != "Darwin":
        return
    try:
        from AppKit import NSApplication

        for win in NSApplication.sharedApplication().windows():
            win.setTabbingMode_(2)  # NSWindowTabbingModeDisallowed = 2
    except Exception:
        pass


def get_resource_path(relative_path: str) -> str:
    """Get the path to a bundled resource, works for dev and PyInstaller."""
    import sys

    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        base_path = Path(getattr(sys, "_MEIPASS", ""))
    else:
        # Running in development
        base_path = Path(__file__).parent
    return str(base_path / relative_path)


def start(
    pipeline: "Pipeline",
    debug: bool = False,
    inputs=(),
    max_pixels: int = 4096,
):
    """
    Create and start the pywebview window with the API.

    Args:
        pipeline: The live pipeline whose graph the GUI displays.
        debug: Enable developer tools
        inputs: Optional result files/folders to populate the sidebar at startup.
        max_pixels: Maximum image/mask preview dimension in pixels.
    """
    _set_macos_app_name(pipeline.name)
    api = GUIBackend(pipeline, inputs=inputs, max_pixels=max_pixels)
    html_path = get_resource_path("index.html")
    window = webview.create_window(
        title=pipeline.name,
        url=html_path,
        js_api=api,
        width=1200,
        height=800,
        min_size=(800, 600),
    )

    def on_started():
        api.set_window(window)
        _disable_macos_tabbing()
        arrange_native_menu()

    from .native_menu import arrange_native_menu, build_native_menu

    webview.start(func=on_started, debug=debug, menu=build_native_menu(window))
    if api._restart_requested:
        import os
        import sys

        arguments = [sys.executable, "-m", "collectra.main", "--pipeline",
                     str(Path(pipeline.path).resolve() / "pipeline.yaml"), "gui"]
        if debug:
            arguments.append("--debug")
        arguments.extend(["--max-pixels", str(max_pixels)])
        # Reuse the environment and process. Startup restores saved results,
        # the selected page, and layout rather than the original launch inputs.
        os.execv(sys.executable, arguments)
    return window
