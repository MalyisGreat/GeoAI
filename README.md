# GeoBot Superhuman

`GeoBot Superhuman` is a visual geolocation training stack aimed at GeoGuessr-style inference without relying on proprietary image APIs.

The project is built around two data paths:

- `OSV-5M` for full-scale global training with direct public downloads from Hugging Face dataset files.
- `Geograph sample` for tiny local smoke tests that only download a few hundred kilobytes.

## Core model

The default architecture is now an AtlasMoE-style base stack:

- shared vision trunk
- coarse geocell router
- region-specialist mixture-of-experts
- explicit clue heads for land cover, climate, soil, drive side, road index, and distance-to-sea proxy
- probabilistic globe head with multi-hypothesis location output
- retrieval head plus clue-aware candidate reranking

## Why this structure

- It scales up to H100/A100 training by enabling mixed precision, expert routing, probabilistic training losses, and optional `timm` backbones.
- It still runs a full smoke loop on CPU or a small consumer GPU.
- It keeps downloads legal and reproducible by using open datasets with direct shard sync and zip streaming.

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

## H100 full run

Use the H100 AtlasMoE config and fast sync:

```bash
python -m pip install -e ".[accelerated]"
python scripts/a100_parallel_download.py --config configs/h100_atlas.yaml --splits train --max-workers 32 --log-dir logs/h100-sync
python scripts/train.py --config configs/h100_atlas.yaml
```

Or run the Linux orchestrator script:

```bash
bash scripts/a100_full_run.sh
```

Useful logs:

- `logs/a100/<timestamp>/download-console.log`
- `logs/a100/<timestamp>/train-console.log`
- `logs/a100/<timestamp>/download/sync.jsonl`
- `runs/atlasmoe-h100/metrics/train.jsonl`
- `runs/atlasmoe-h100/metrics/train_steps.jsonl`
- `runs/atlasmoe-h100/metrics/val.jsonl`
- `runs/atlasmoe-h100/artifacts/startup.json`

## Open data used

- `OSV-5M`: public visual geolocation dataset hosted on Hugging Face.
- `Geograph sample`: public Creative Commons geotagged image sample hosted by Geograph.

## Important note

This repository builds the geolocation model and evaluation pipeline. It does not automate any game website or browser interaction.
