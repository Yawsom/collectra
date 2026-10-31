#!/usr/bin/env python3
"""Report overlapping ImageCrop bounding boxes in Collectra results files.

Each argument can be a ``results.yaml`` file, a result folder (such as
``Page022.livy``), or a parent folder, which is searched for every
``results.yaml`` below it. Crops are loaded with Collectra's own artefact
classes, so every node holding ``collectra.ImageCrop`` items is checked, and
only crops of the same image are compared with each other.

Boxes are compared in relative coordinates, so the numbers do not depend on
image size. The table lists each pair by artefact id with their IoU
(intersection area over union area). ``--json`` gives the full detail for each
pair: node names, the share of the smaller box that overlaps (so a box sitting
entirely inside another is 1.0), and the overlap in relative and pixel terms.

Boxes that merely touch along an edge are not overlaps. Tiny slivers, such as
rounding noise between boxes meant to abut, are ignored below ``--min-overlap``.

With ``--remove-smaller`` the smaller box of each overlapping pair is deleted
from its results file (``--dry-run`` shows what would go without writing).
"Smaller" means smaller area; for equal areas the lower-confidence crop goes,
then the one with the later id. Pairs are resolved biggest overlap first, so
when A overlaps B and B overlaps C, no overlap is left among the crops kept.
Every option that narrows the pairs, such as ``--same-node-only``, narrows what
is removed too. A crop that other items name as a parent (text read from it, a
link to it) is not removed, because that would orphan them; it is listed
instead. With ``--repoint-dependents`` such a crop is removed anyway and the
items that named it as a parent are pointed at the crop that stays. That is
only sound when the two boxes are nearly the same, so it is limited to pairs
whose IoU reaches ``--repoint-min-iou`` (default 0.9); a chain of removals
carries the dependents to the final survivor. The file is rewritten with Collectra's own YAML writer, and a node left
with no crops stays in the file as an empty list.

Examples:
    scripts/find_overlapping_crops.py Page022.livy
    scripts/find_overlapping_crops.py /path/to/output --same-node-only
    scripts/find_overlapping_crops.py /path/to/output --min-overlap 0.2 --json
    scripts/find_overlapping_crops.py /path/to/output --remove-smaller --dry-run
    scripts/find_overlapping_crops.py /path/to/output --remove-smaller --repoint-dependents
    scripts/find_overlapping_crops.py /path/to/output --remove-smaller --same-node-only

Exit status: 0 when no overlap is left (none found, or all removed), 1 when
overlaps are found (or remain because crops could not be removed), 2 when no
results files were found.
"""

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from collectra.types.base import ArtefactNode
from collectra.types.images import ImageCrop
from collectra.utils import write_yaml

METADATA_KEY = "collectra_results_metadata"
CROP_TYPE = "collectra.ImageCrop"
EPSILON = 1e-9
DEFAULT_MIN_OVERLAP = 0.005

Box = tuple[float, float, float, float]


class Overlap(BaseModel):
    """Two crops of the same image whose boxes overlap."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    a: ImageCrop
    b: ImageCrop
    intersection: Box
    iou: float
    share_of_smaller: float


def find_results_files(paths: list[Path]) -> list[Path]:
    """Expand files, result folders and parent folders into results.yaml files."""
    found: list[Path] = []
    for path in map(Path, paths):
        if path.is_file():
            found.append(path)
        elif (path / "results.yaml").is_file():
            found.append(path / "results.yaml")
        elif path.is_dir():
            found.extend(sorted(path.rglob("results.yaml")))
    return found


@contextlib.contextmanager
def _quiet_collectra():
    """Collectra logs a traceback for every item it cannot load; we count them."""
    logger = logging.getLogger("collectra")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(previous)


def load_crops(results_file: Path) -> tuple[list[ImageCrop], int]:
    """Load every ImageCrop in a results file; also count those that failed to load."""
    results_file = Path(results_file)
    with open(results_file) as stream:
        data = yaml.safe_load(stream) or {}
    expected: dict[str, int] = {}
    for key, value in data.items():
        if key == METADATA_KEY:
            continue
        entries = value if isinstance(value, list) else [value]
        count = sum(
            1 for e in entries if isinstance(e, dict) and e.get("type") == CROP_TYPE
        )
        if count:
            expected[key] = count

    crops: list[ImageCrop] = []
    with _quiet_collectra():
        for key in expected:
            node = ArtefactNode(key, types={ImageCrop})
            node.process(key, results_file.parent)
            # Links stand in for the crop they point to; compare the crop itself.
            crops.extend(i for i in node.items.values() if type(i) is ImageCrop)
    return crops, sum(expected.values()) - len(crops)


def relative_box(crop: ImageCrop) -> Box:
    """The crop's corners (left, top, right, bottom) as fractions of the image."""
    half_w, half_h = crop.width_relative / 2, crop.height_relative / 2
    return (
        max(0.0, crop.x_center - half_w),
        max(0.0, crop.y_center - half_h),
        min(1.0, crop.x_center + half_w),
        min(1.0, crop.y_center + half_h),
    )


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_between(a: ImageCrop, b: ImageCrop) -> Overlap | None:
    """The overlap of two crops' boxes, or None if they do not share any area."""
    box_a, box_b = relative_box(a), relative_box(b)
    inter: Box = (
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    )
    shared = _area(inter)
    if shared <= EPSILON:
        return None
    area_a, area_b = _area(box_a), _area(box_b)
    smaller = min(area_a, area_b)
    return Overlap(
        a=a,
        b=b,
        intersection=inter,
        iou=shared / (area_a + area_b - shared),
        share_of_smaller=shared / smaller if smaller > 0 else 1.0,
    )


def find_overlaps(
    crops: list[ImageCrop],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    min_iou: float = 0.0,
    same_node_only: bool = False,
    ignore_lineage: bool = False,
) -> list[Overlap]:
    """Every overlapping pair of crops of the same image, biggest overlap first."""
    by_image: dict[str, list[ImageCrop]] = {}
    for crop in crops:
        by_image.setdefault(str(crop.get_path().resolve()), []).append(crop)

    found: list[Overlap] = []
    for group in by_image.values():
        for index, a in enumerate(group):
            for b in group[index + 1 :]:
                if same_node_only and a.name != b.name:
                    continue
                if ignore_lineage and (a.id in b.parents or b.id in a.parents):
                    continue
                overlap = overlap_between(a, b)
                if (
                    overlap
                    and overlap.share_of_smaller >= min_overlap
                    and overlap.iou >= min_iou
                ):
                    found.append(overlap)
    return sorted(found, key=lambda o: (-o.share_of_smaller, -o.iou, o.a.id, o.b.id))


class Removal(BaseModel):
    """The crop to delete from an overlapping pair, and the one that stays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    removed: ImageCrop
    kept: ImageCrop
    overlap: Overlap
    blocked_by: list[str] = []
    repointed: list[str] = []
    note: str = ""
    applied: bool = False


def choose_smaller(overlap: Overlap) -> tuple[ImageCrop, ImageCrop]:
    """Return ``(crop to remove, crop to keep)``: the smaller area goes.

    Equal areas are decided by confidence (a crop with none counts as lowest),
    then by id, so the same input always gives the same answer.
    """
    a, b = overlap.a, overlap.b
    area_a, area_b = _area(relative_box(a)), _area(relative_box(b))
    if abs(area_a - area_b) > EPSILON:
        return (a, b) if area_a < area_b else (b, a)
    confidence_a = -1.0 if a.confidence is None else a.confidence
    confidence_b = -1.0 if b.confidence is None else b.confidence
    if confidence_a != confidence_b:
        return (a, b) if confidence_a < confidence_b else (b, a)
    return (a, b) if a.id > b.id else (b, a)


def dependents_map(data: dict) -> dict[str, set[str]]:
    """For each id, the ids of the items that list it as a parent."""
    found: dict[str, set[str]] = {}
    for key, value in data.items():
        if key == METADATA_KEY:
            continue
        for entry in value if isinstance(value, list) else [value]:
            if not isinstance(entry, dict) or not entry.get("parents"):
                continue
            parents = entry["parents"]
            for parent in parents if isinstance(parents, list) else [parents]:
                found.setdefault(str(parent), set()).add(str(entry.get("id", key)))
    return found


def plan_removals(
    overlaps: list[Overlap],
    dependents: dict[str, set[str]] | None = None,
    repoint_min_iou: float | None = None,
) -> list[Removal]:
    """Decide which crops to remove so no two kept crops overlap.

    Overlaps are resolved in the order given (biggest first). A pair with a
    crop that is already gone is already resolved. A crop that something else
    depends on is not removed, unless everything depending on it is itself
    being removed, or ``repoint_min_iou`` is set and the pair's IoU reaches it:
    then the dependents move to the crop that stays (``repointed``). Pairs left
    blocked come back with ``blocked_by`` set. Passes repeat because removing a
    dependent can unblock its parent.
    """
    dependents = {key: set(value) for key, value in (dependents or {}).items()}
    gone: set[str] = set()
    removals: dict[tuple[str, str], Removal] = {}
    blocked: dict[tuple[str, str], Removal] = {}
    changed = True
    while changed:
        changed = False
        for overlap in overlaps:
            if overlap.a.id in gone or overlap.b.id in gone:
                continue
            removed, kept = choose_smaller(overlap)
            holding = sorted(dependents.get(removed.id, set()) - gone)
            pair = (overlap.a.id, overlap.b.id)
            can_repoint = repoint_min_iou is not None and overlap.iou >= repoint_min_iou
            if holding and not can_repoint:
                note = ""
                if repoint_min_iou is not None:
                    note = (
                        f"IoU {overlap.iou:.1%} is below the re-point minimum "
                        f"of {repoint_min_iou:.0%}"
                    )
                blocked[pair] = Removal(
                    removed=removed,
                    kept=kept,
                    overlap=overlap,
                    blocked_by=holding,
                    note=note,
                )
                continue
            blocked.pop(pair, None)
            gone.add(removed.id)
            if holding:
                # They now hang on the crop that stays, in case that goes later.
                dependents.setdefault(kept.id, set()).update(holding)
            removals[pair] = Removal(
                removed=removed, kept=kept, overlap=overlap, repointed=holding
            )
            changed = True
    still_blocked = [
        r
        for r in blocked.values()
        if r.removed.id not in gone and r.kept.id not in gone
    ]
    return [*removals.values(), *still_blocked]


def apply_removals(results_file: Path, data: dict, removals: list[Removal]) -> None:
    """Delete the removed crops from ``data`` and write the results file back.

    Items that named a removed crop as a parent are pointed at the crop that
    replaces it (following a chain of replacements to the final survivor).
    """
    doomed = {(r.removed.name, r.removed.id) for r in removals}
    redirect = {r.removed.id: r.kept.id for r in removals if r.repointed}

    def survivor(crop_id: str) -> str:
        seen: set[str] = set()
        while crop_id in redirect and crop_id not in seen:
            seen.add(crop_id)
            crop_id = redirect[crop_id]
        return crop_id

    if redirect:
        for key, value in data.items():
            if key == METADATA_KEY:
                continue
            for entry in value if isinstance(value, list) else [value]:
                parents = entry.get("parents") if isinstance(entry, dict) else None
                if not parents:
                    continue
                if isinstance(parents, list):
                    moved = [survivor(str(p)) for p in parents]
                    entry["parents"] = list(dict.fromkeys(moved))
                else:
                    entry["parents"] = survivor(str(parents))

    for key in {name for name, _ in doomed}:
        value = data[key]
        entries = value if isinstance(value, list) else [value]
        left = [e for e in entries if (key, e.get("id")) not in doomed]
        # Collectra writes one item as a mapping, several as a list, none as [].
        data[key] = left[0] if len(left) == 1 else left
    write_yaml(data, results_file)


def _pixels(overlap: Overlap) -> tuple[int, int, int, int]:
    left, top, right, bottom = overlap.intersection
    width, height = overlap.a.raw_width, overlap.a.raw_height
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )


def _label(crop: ImageCrop) -> str:
    return crop.id


def _record(
    results_file: Path, overlap: Overlap, removal: Removal | None = None
) -> dict:
    return {
        "file": str(results_file),
        "image": str(overlap.a.get_path()),
        "a": {"node": overlap.a.name, "id": overlap.a.id},
        "b": {"node": overlap.b.name, "id": overlap.b.id},
        "iou": overlap.iou,
        "share_of_smaller": overlap.share_of_smaller,
        "intersection": list(overlap.intersection),
        "intersection_pixels": list(_pixels(overlap)),
        "removed": (
            {"node": removal.removed.name, "id": removal.removed.id}
            if removal
            else None
        ),
        "kept": (
            {"node": removal.kept.name, "id": removal.kept.id} if removal else None
        ),
        "blocked_by": removal.blocked_by if removal else [],
        "repointed": removal.repointed if removal else [],
        "applied": bool(removal and removal.applied),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report overlapping ImageCrop bounding boxes in results files.",
        epilog="Exit status: 0 = no overlaps, 1 = overlaps found, 2 = no results files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="results.yaml files, result folders, or folders containing them",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP,
        help="ignore overlaps smaller than this fraction of the smaller box "
        f"(default {DEFAULT_MIN_OVERLAP}, which skips rounding slivers)",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.0,
        help="ignore overlaps with a lower intersection-over-union",
    )
    parser.add_argument(
        "--same-node-only",
        action="store_true",
        help="only compare crops that belong to the same node",
    )
    parser.add_argument(
        "--ignore-lineage",
        action="store_true",
        help="skip pairs where one crop lists the other as a parent",
    )
    parser.add_argument(
        "--remove-smaller",
        action="store_true",
        help="delete the smaller crop of each overlapping pair from its results "
        "file (crops that other items depend on are kept and listed)",
    )
    parser.add_argument(
        "--repoint-dependents",
        action="store_true",
        help="with --remove-smaller: also remove crops that other items depend "
        "on, pointing those items at the crop that stays (only for near-"
        "duplicates, see --repoint-min-iou)",
    )
    parser.add_argument(
        "--repoint-min-iou",
        type=float,
        default=0.9,
        help="with --repoint-dependents: the lowest IoU at which dependents are "
        "moved to the kept crop (default 0.9); below it the crop is not removed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --remove-smaller: show what would be removed, change nothing",
    )
    parser.add_argument("--json", action="store_true", help="print JSON, not a table")
    return parser


def _report_overlaps(console: Console, records: list[tuple[Path, Overlap]], total: int):
    table = Table(title="Overlapping image crops")
    for column, justify in [
        ("Results", "left"),
        ("Crop A", "left"),
        ("Crop B", "left"),
        ("IoU", "right"),
    ]:
        table.add_column(column, justify=justify)
    for results_file, overlap in records:
        table.add_row(
            results_file.parent.name,
            _label(overlap.a),
            _label(overlap.b),
            f"{overlap.iou:.1%}",
        )
    console.print(table)
    affected = len({f for f, _ in records})
    console.print(
        f"{len(records)} overlapping pair(s) in {affected} of {total} results file(s)."
    )


def _report_removals(
    console: Console, plans: list[tuple[Path, Removal]], total: int, dry_run: bool
):
    done = [(f, r) for f, r in plans if not r.blocked_by]
    blocked = [(f, r) for f, r in plans if r.blocked_by]
    if done:
        table = Table(title="Would remove" if dry_run else "Removed image crops")
        columns = [
            ("Results", "left"),
            ("Removed (smaller)", "left"),
            ("Kept", "left"),
            ("IoU", "right"),
        ]
        moved = any(plan.repointed for _, plan in done)
        if moved:
            columns.append(("Re-pointed to kept", "left"))
        for column, justify in columns:
            table.add_column(column, justify=justify)
        for results_file, plan in done:
            row = [
                results_file.parent.name,
                _label(plan.removed),
                _label(plan.kept),
                f"{plan.overlap.iou:.1%}",
            ]
            if moved:
                row.append(", ".join(plan.repointed))
            table.add_row(*row)
        console.print(table)
        verb = "Would remove" if dry_run else "Removed"
        files = len({f for f, _ in done})
        console.print(
            f"{verb} {len(done)} crop(s) from {files} of {total} results file(s)."
        )
    if blocked:
        table = Table(title="Not removed: other items depend on these crops")
        columns = ["Results", "Crop", "Overlaps", "Depended on by"]
        noted = any(plan.note for _, plan in blocked)
        if noted:
            columns.append("Why")
        for column in columns:
            table.add_column(column)
        for results_file, plan in blocked:
            row = [
                results_file.parent.name,
                _label(plan.removed),
                _label(plan.kept),
                ", ".join(plan.blocked_by),
            ]
            if noted:
                row.append(plan.note)
            table.add_row(*row)
        console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and not args.remove_smaller:
        parser.error("--dry-run only makes sense with --remove-smaller")
    if args.repoint_dependents and not args.remove_smaller:
        parser.error("--repoint-dependents only makes sense with --remove-smaller")
    if args.repoint_min_iou != 0.9 and not args.repoint_dependents:
        parser.error("--repoint-min-iou only makes sense with --repoint-dependents")
    results_files = find_results_files(args.paths)
    if not results_files:
        print("No results.yaml files found.", file=sys.stderr)
        return 2

    overlaps_found: list[tuple[Path, Overlap]] = []
    plans: list[tuple[Path, Removal]] = []
    unloaded: dict[Path, int] = {}
    for results_file in results_files:
        crops, skipped = load_crops(results_file)
        if skipped:
            unloaded[results_file] = skipped
        overlaps = find_overlaps(
            crops,
            args.min_overlap,
            args.min_iou,
            args.same_node_only,
            args.ignore_lineage,
        )
        if not args.remove_smaller:
            overlaps_found.extend((results_file, o) for o in overlaps)
            continue
        if not overlaps:
            continue
        with open(results_file) as stream:
            data = yaml.safe_load(stream) or {}
        file_plans = plan_removals(
            overlaps,
            dependents_map(data),
            args.repoint_min_iou if args.repoint_dependents else None,
        )
        doomed = [p for p in file_plans if not p.blocked_by]
        if doomed and not args.dry_run:
            apply_removals(results_file, data, doomed)
            for plan in doomed:
                plan.applied = True
        plans.extend((results_file, p) for p in file_plans)

    if args.remove_smaller:
        # Left over: everything under --dry-run, or what could not be removed.
        remaining = [p for _, p in plans if args.dry_run or p.blocked_by]
    else:
        remaining = overlaps_found

    if args.json:
        if args.remove_smaller:
            payload = [_record(f, p.overlap, p) for f, p in plans]
        else:
            payload = [_record(f, o) for f, o in overlaps_found]
        print(json.dumps(payload, indent=2))
        return 1 if remaining else 0

    console = Console(width=None if sys.stdout.isatty() else 160)
    if args.remove_smaller and plans:
        _report_removals(console, plans, len(results_files), args.dry_run)
    elif overlaps_found:
        _report_overlaps(console, overlaps_found, len(results_files))
    else:
        console.print(
            f"No overlapping image crops found in {len(results_files)} results file(s)."
        )
    if unloaded:
        total = sum(unloaded.values())
        console.print(
            f"[yellow]{total} crop(s) in {len(unloaded)} file(s) could not be "
            "loaded (missing image or invalid record) and were not checked.[/yellow]"
        )
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
