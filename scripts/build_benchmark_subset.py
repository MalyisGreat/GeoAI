from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MISSING_VALUE = "__missing__"


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def build_metadata_index(path: Path | None) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    if path is None or not path.exists():
        return {}, {}

    by_image_path: dict[str, dict[str, object]] = {}
    by_image_id: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = str(row.get("image_path", "")).replace("\\", "/")
            if image_path:
                by_image_path[image_path] = row
            image_id = row.get("image_id")
            if image_id is not None:
                by_image_id[str(image_id)] = row
    return by_image_path, by_image_id


def has_column_in_index(index: dict[str, dict[str, object]], column: str) -> bool:
    for row in index.values():
        if column in row:
            return True
    return False


def lookup_metadata(
    row: dict[str, str],
    by_image_path: dict[str, dict[str, object]],
    by_image_id: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    image_path = str(row.get("image_path", "")).replace("\\", "/")
    if image_path in by_image_path:
        return by_image_path[image_path]
    image_id = Path(image_path).stem
    return by_image_id.get(image_id)


def stable_score(seed: int, split: str, key: str) -> int:
    payload = f"{seed}:{split}:{key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def row_key(row: dict[str, str]) -> str:
    image_path = row.get("image_path")
    if image_path:
        return image_path.replace("\\", "/")
    return json.dumps(row, sort_keys=True)


def stratify_value(
    row: dict[str, str],
    metadata_row: dict[str, object] | None,
    column: str | None,
) -> str:
    if column is None:
        return "__all__"
    if metadata_row is not None and column in metadata_row:
        value = metadata_row[column]
    else:
        value = row.get(column)
    if value in (None, ""):
        return MISSING_VALUE
    return str(value)


def allocate_quotas(group_sizes: dict[str, int], limit: int) -> dict[str, int]:
    if limit <= 0 or not group_sizes:
        return {group: 0 for group in group_sizes}

    total = sum(group_sizes.values())
    if limit >= total:
        return dict(group_sizes)

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for group, size in group_sizes.items():
        raw_quota = (size * limit) / total
        base_quota = min(size, int(raw_quota))
        quotas[group] = base_quota
        assigned += base_quota
        remainders.append((raw_quota - base_quota, group))

    remaining = limit - assigned
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, group in remainders:
        if remaining <= 0:
            break
        if quotas[group] >= group_sizes[group]:
            continue
        quotas[group] += 1
        remaining -= 1
    return quotas


def select_rows(
    rows: list[tuple[dict[str, str], str]],
    limit: int,
    seed: int,
    split: str,
) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda item: stable_score(seed, split, row_key(item[0])))
    return [row for row, _group in ranked[:limit]]


def stratified_select(
    rows: list[tuple[dict[str, str], str]],
    limit: int,
    seed: int,
    split: str,
) -> list[dict[str, str]]:
    if limit <= 0 or not rows:
        return []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row, group in rows:
        grouped[group].append(row)

    group_sizes = {group: len(group_rows) for group, group_rows in grouped.items()}
    quotas = allocate_quotas(group_sizes, min(limit, len(rows)))

    selected: list[dict[str, str]] = []
    leftovers: list[tuple[dict[str, str], str]] = []
    for group, group_rows in grouped.items():
        ranked = sorted(group_rows, key=lambda row: stable_score(seed, split, row_key(row)))
        quota = quotas[group]
        selected.extend(ranked[:quota])
        leftovers.extend((row, group) for row in ranked[quota:])

    if len(selected) < min(limit, len(rows)):
        needed = min(limit, len(rows)) - len(selected)
        leftovers.sort(key=lambda item: stable_score(seed, split, row_key(item[0])))
        selected.extend(row for row, _group in leftovers[:needed])

    selected.sort(key=lambda row: stable_score(seed, split, row_key(row)))
    return selected


def build_subset(
    manifest_rows: list[dict[str, str]],
    metadata_index: tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]],
    train_limit: int,
    val_limit: int,
    seed: int,
    stratify_column: str | None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    by_image_path, by_image_id = metadata_index
    split_rows: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    stratify_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in manifest_rows:
        split = row.get("split", "unknown")
        metadata_row = lookup_metadata(row, by_image_path, by_image_id)
        group = stratify_value(row, metadata_row, stratify_column)
        split_rows[split].append((row, group))
        stratify_counts[split][group] += 1

    selected: list[dict[str, str]] = []
    selected_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split, rows in split_rows.items():
        if split == "train":
            limit = min(train_limit, len(rows))
        elif split == "val":
            limit = min(val_limit, len(rows))
        else:
            limit = len(rows)

        if stratify_column is None:
            chosen = select_rows(rows, limit, seed, split)
        else:
            chosen = stratified_select(rows, limit, seed, split)
        selected.extend(chosen)

        if stratify_column is not None:
            selected_set = {row_key(row) for row in chosen}
            for row, group in rows:
                if row_key(row) in selected_set:
                    selected_counts[split][group] += 1

    selected.sort(key=lambda row: (row.get("split", ""), stable_score(seed, row.get("split", ""), row_key(row))))
    return selected, {
        "available": {split: dict(counts) for split, counts in stratify_counts.items()},
        "selected": {split: dict(counts) for split, counts in selected_counts.items()},
    }


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic benchmark subset manifest.")
    parser.add_argument("--input-manifest", required=True, help="Source manifest CSV")
    parser.add_argument("--output-manifest", required=True, help="Destination manifest CSV")
    parser.add_argument("--train-limit", type=int, default=10000, help="Maximum train rows to keep")
    parser.add_argument("--val-limit", type=int, default=2000, help="Maximum val rows to keep")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument(
        "--metadata-jsonl",
        default=None,
        help="Optional sidecar metadata JSONL used for stratification. Defaults to sibling metadata.jsonl if present.",
    )
    parser.add_argument(
        "--stratify-column",
        default=None,
        help="Optional column name used to preserve proportions. Can come from manifest or metadata JSONL.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_manifest = resolve_path(args.input_manifest)
    output_manifest = resolve_path(args.output_manifest)
    metadata_path = Path(args.metadata_jsonl) if args.metadata_jsonl else input_manifest.with_name("metadata.jsonl")
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    if not metadata_path.exists():
        metadata_path = None

    fieldnames, manifest_rows = load_manifest(input_manifest)
    metadata_index = build_metadata_index(metadata_path)

    if args.stratify_column is not None:
        manifest_has_column = args.stratify_column in fieldnames
        metadata_has_column = any(has_column_in_index(index, args.stratify_column) for index in metadata_index)
        if not manifest_has_column and not metadata_has_column:
            raise KeyError(
                f"Stratify column '{args.stratify_column}' not found in manifest columns "
                f"or metadata sidecar {metadata_path}."
            )

    selected_rows, stratify_summary = build_subset(
        manifest_rows=manifest_rows,
        metadata_index=metadata_index,
        train_limit=max(0, args.train_limit),
        val_limit=max(0, args.val_limit),
        seed=args.seed,
        stratify_column=args.stratify_column,
    )
    write_manifest(output_manifest, fieldnames, selected_rows)

    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "metadata_jsonl": str(metadata_path) if metadata_path is not None else None,
        "seed": args.seed,
        "stratify_column": args.stratify_column,
        "rows": len(selected_rows),
        "train_rows": sum(1 for row in selected_rows if row.get("split") == "train"),
        "val_rows": sum(1 for row in selected_rows if row.get("split") == "val"),
        "stratify": stratify_summary if args.stratify_column is not None else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
