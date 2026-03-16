# Architecture Plan

## Project

- Name: GeoBot Superhuman
- Date started: 2026-03-16
- Owner: Josh + Codex
- Status: active
- Scope: Build an open-data visual geolocation training and inference stack that uses explicit clue supervision, region-specialist experts, probabilistic globe modeling, candidate reranking, and fast shard-streamed data access while remaining locally smoke-testable on the current Windows laptop.

## Vision and Context

- Problem statement: Train a strong GeoGuessr-style model without depending on paid or proprietary image APIs and without making local verification prohibitively expensive.
- In-scope:
  - public dataset downloaders
  - manifest generation
  - AtlasMoE-style hybrid geolocation model
  - training, evaluation, retrieval reranking, and benchmarking
  - structured metrics and smoke tests
- Out-of-scope:
  - browser automation against live game sites
  - closed-source image providers
  - large-scale distributed orchestration beyond single-node assumptions
- Primary assumptions:
  - full-scale runs happen on Linux or Windows with access to at least one A100-class GPU
  - local smoke runs happen on this machine, which currently exposes CPU-only PyTorch even though an RTX 3060 is installed
  - public HTTP downloads are acceptable, but smoke tests must stay small
- Constraints:
  - no paid image API
  - local smoke test should download only a tiny sample
  - code should run with the currently installed Python 3.13 environment

## Architecture Overview

- Components and boundaries:
  - `src/geobot/data`: dataset providers, downloaders, manifest builders, dataset objects
  - `src/geobot/model`: backbone selection, geocell router, region experts, clue heads, probabilistic globe head, and retrieval head
  - `src/geobot/train`: training loop, checkpointing, and evaluation orchestration
  - `src/geobot/eval`: retrieval reranking and metric aggregation
  - `src/geobot/utils`: configuration, geo math, logging, and IO helpers
  - `scripts/`: operator entrypoints
  - `benchmarks/`: repeatable throughput benchmark
- Responsibility map:
  - providers know how to fetch and normalize source data
  - manifests are the source of truth for labels and file references
  - model code never reaches out to the network
  - training code owns optimization, metrics, checkpoints, and resource snapshots
- Data flow:
  - public dataset -> local raw files or zip shards -> normalized manifest -> zip member index -> dataset loader -> router + experts + clue heads + probabilistic head -> checkpoint + metrics -> retrieval + clue-aware reranking -> evaluation report
- External dependencies:
  - Hugging Face static dataset files for `OSV-5M`
  - Geograph static sample zip
  - PyTorch, pandas, numpy, Pillow
  - optional `timm` and `psutil` for higher-end training and runtime stats

## Technical Design

- Language/framework stack: Python, PyTorch, pandas, YAML config, pytest
- Core interfaces:
  - provider classes expose `prepare(...)`
  - manifests are parquet or CSV with normalized image paths plus clue metadata
  - model forward returns routed embeddings, probabilistic location parameters, clue predictions, and retrieval logits
  - evaluation consumes model outputs plus gallery embeddings and per-cell clue priors
- Persistence/storage model:
  - raw downloads under `data/<provider>/raw`
  - manifests under `data/<provider>/manifests`
  - runs under `runs/<experiment>`
- State and consistency model:
  - manifests are immutable run inputs once training starts
  - checkpoints embed config + label vocab so inference is reproducible
  - metrics are append-only JSONL plus a summary JSON
- Deployment topology:
  - local smoke: single-process CPU or small GPU
  - full training: single-node H100/A100 with larger batch size, BF16, zip-streamed shards, and optional compiled model path
- Non-functional requirements (availability, latency, scale, cost):
  - smoke verification under a few minutes
  - A100 path should support large batches with mixed precision
  - downloads must tolerate interruption and reuse local cache
  - metrics must capture enough evidence to compare runs

## Decision Log

- Decision ID: ADR-001
- Date: 2026-03-16
- Owner: Codex
- Decision: Use `OSV-5M` as the primary global dataset and `Geograph sample` as the smoke-test dataset.
- Rationale: `OSV-5M` is a strong open benchmark for street-view geolocation, while Geograph provides tiny, legal, direct-download tests.
- Alternatives considered: scraping random Wikimedia pages, stale thumbnail URLs from dataset metadata
- Risks: `OSV-5M` image shards are large and unsuitable for local smoke tests.
- Status: active

- Decision ID: ADR-002
- Date: 2026-03-16
- Owner: Codex
- Decision: Use an AtlasMoE-style base stack combining a coarse router, region-specialist experts, explicit clue heads, a probabilistic globe head, retrieval scoring, and clue-aware candidate reranking.
- Rationale: This keeps ambiguity explicit, routes hard cases to specialists, and makes subtle cues trainable instead of hoping a single head absorbs them.
- Alternatives considered: pure classification, pure regression, direct nearest-neighbor only
- Risks: more moving parts, more auxiliary labels, and more tuning knobs.
- Status: active

- Decision ID: ADR-003
- Date: 2026-03-16
- Owner: Codex
- Decision: Make the default smoke backbone a built-in convolutional model and keep `timm` backbones optional.
- Rationale: the current machine has CPU-only PyTorch, so smoke runs must not depend on heavyweight optional installs.
- Alternatives considered: forcing `timm`, forcing torchvision, writing a single tiny-only model
- Risks: smoke accuracy is not representative of the A100 target path.
- Status: active

- Decision ID: ADR-004
- Date: 2026-03-16
- Owner: Codex
- Decision: Prefer shard streaming and zip indexing over eager extraction for full-scale OSV-5M training.
- Rationale: Fast sync plus direct zip reads cuts startup time and peak disk use, which matters more than perfect raw-file convenience on H100-class runs.
- Alternatives considered: full eager extraction, per-image HTTP fetches
- Risks: first-run zip indexing adds some up-front latency and the loader depends on archive integrity.
- Status: active

## Update Cadence

- Trigger: changes in architecture, requirements, or infrastructure
- Review cadence: every material feature addition and before long training jobs
- Who approves updates: project owner
- Last reviewed: 2026-03-16

## Open Risks and Issues

- The local Python environment currently exposes `torch 2.8.0+cpu`; A100 training will require a CUDA-enabled environment.
- `OSV-5M` full image shards are multi-gigabyte zips; large-scale experiments need disk planning and likely Linux for best throughput.
- Smoke data is not globally representative, so only pipeline correctness, not final model quality, is locally verified.

## Change History

- 2026-03-16 | Created initial architecture plan | New project kickoff
- 2026-03-16 | Rebased architecture onto AtlasMoE-style training and zip-streamed data access | Stronger base model and faster H100 startup
- 2026-03-16 | Enabled sequential shard streaming for train batches and byte-level sync telemetry | Higher end-to-end throughput and clearer download planning
