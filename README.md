# GeoBot Superhuman

`GeoBot Superhuman` is a visual geolocation training stack aimed at GeoGuessr-style inference without relying on proprietary image APIs.

The project is built around two data paths:

- `OSV-5M` for full-scale global training with direct public downloads from Hugging Face dataset files.
- `Geograph sample` for tiny local smoke tests that only download a few hundred kilobytes.

## Core model

The default architecture is a hybrid geolocation model:

- image encoder backbone
- coarse geocell classifier
- fine geocell classifier
- spherical coordinate regressor
- gallery retrieval reranker that blends nearest-neighbor evidence with direct regression

That mirrors what works well in open geolocation literature: classification narrows the region, regression sharpens the coordinate, retrieval adds memorization for high-confidence matches.

## Why this structure

- It scales up to A100 training by enabling mixed precision, gradient accumulation, and optional `timm` backbones.
- It still runs a full smoke loop on CPU or a small consumer GPU.
- It keeps downloads legal and reproducible by using open datasets with direct HTTP access.

## Quick start

Install in-place:

```powershell
cd C:\Users\joshj\geoguessr-superhuman
python -m pip install -e .
```

Run the tiny smoke path:

```powershell
python scripts\run_smoke_test.py
```

Download an `OSV-5M` metadata sample without images:

```powershell
python scripts\download_dataset.py --config configs\base.yaml --split train --max-rows 2048
```

Run training with an existing manifest:

```powershell
python scripts\train.py --config configs\smoke.yaml
```

Run a small benchmark:

```powershell
python benchmarks\run.py --config configs\smoke.yaml --output benchmarks\latest-metrics.json
```

## A100 full run

Use the A100 config and parallel downloader:

```bash
python -m pip install -e ".[accelerated]"
python scripts/a100_parallel_download.py --config configs/a100_full.yaml --max-workers 6 --log-dir logs/a100-download
python scripts/train.py --config configs/a100_full.yaml
```

Or run the Linux orchestrator script:

```bash
bash scripts/a100_full_run.sh
```

Useful logs:

- `logs/a100/<timestamp>/download-console.log`
- `logs/a100/<timestamp>/train-console.log`
- `logs/a100/<timestamp>/download/download.jsonl`
- `runs/geo-superhuman-a100-full/metrics/train.jsonl`
- `runs/geo-superhuman-a100-full/metrics/val.jsonl`

## Open data used

- `OSV-5M`: public visual geolocation dataset hosted on Hugging Face.
- `Geograph sample`: public Creative Commons geotagged image sample hosted by Geograph.

## Important note

This repository builds the geolocation model and evaluation pipeline. It does not automate any game website or browser interaction.
