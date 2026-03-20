# GeoAI

GeoAI is an open geolocation training stack for offline image-to-location modeling. The current default recipe uses a hierarchical head with `country -> region -> geocell -> local offset` prediction on top of a stronger `ConvNeXt Small` backbone, with open-data OSV5M subset downloads and detailed JSON logging for both download and training.

What this repository is for:
- Downloading a reproducible open geolocation subset from OSV5M.
- Training a hierarchical geolocation model on one GPU.
- Logging every important stage of the run to timestamped files.
- Running controlled local benchmarks and architecture sweeps.

What this repository is not for:
- Automating any live game website.
- Bypassing rate limits or anti-bot systems.
- Interacting with GeoGuessr directly.

**Quick Start**
For a clean A100 run on Linux:

```bash
git clone https://github.com/MalyisGreat/GeoAI.git
cd GeoAI
bash scripts/a100_full_run.sh
```

That command:
- installs the repo in editable mode
- downloads a 200k-image OSV5M subset
- writes a resolved run config for the specific launch
- starts hierarchical training with structured console logs

The default A100 run uses:
- backbone: `convnext_small`
- image size: `224`
- hierarchy: `country`, `region`, `geocell`, `local offset`
- data target: `200,000` images (`4` shards x `50,000`)

**Direct Commands**
If you want to drive the steps yourself:

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_osv5m_subset.py --shards 00 01 02 03 --limit-per-shard 50000 --output-dir data/osv5m_200k_a100
python scripts/train_once.py configs/autonomy/osv5m_200k_convnext_small_hierarchical_224_a100.yaml
```

If you want a larger subset on the same A100 launcher:

```bash
bash scripts/a100_full_run.sh --shards 00 01 02 03 04 05 06 07 08 09 --output-dir data/osv5m_500k_a100 --max-train-samples 450000 --max-val-samples 50000
```

**Logs**
Each A100 launch writes a timestamped log directory under `logs/a100/<timestamp>`.

Important files:
- `logs/a100/<timestamp>/startup.json`
- `logs/a100/<timestamp>/commands.json`
- `logs/a100/<timestamp>/download-console.log`
- `logs/a100/<timestamp>/train-console.log`
- `logs/a100/<timestamp>/resolved-config.yaml`

Training artifacts land under `runs/a100/<timestamp>/`.

Important files:
- `runs/a100/<timestamp>/cycle-00-bootstrap/history.json`
- `runs/a100/<timestamp>/cycle-00-bootstrap/best.pt`

Training emits structured JSON logs for:
- `train_start`
- `epoch_start`
- `train_step`
- `periodic`
- `epoch_end`

The periodic and epoch-end events include validation metrics such as:
- `median_km`
- `within_100km`
- `accepted`

**Repo Layout**
- `src/geo_autolab`: core model, training, eval, and autonomy code
- `configs/model`: model definitions
- `configs/autonomy`: train/eval run configs
- `scripts/prepare_osv5m_subset.py`: open-data subset download and manifest build
- `scripts/a100_prepare_and_train.py`: one-command A100 orchestrator with logging
- `scripts/train_once.py`: launch one configured experiment

**Open Data**
- OSV5M from [Hugging Face](https://huggingface.co/datasets/osv5m/osv5m)
- Small bootstrap path from KartaView and Wikimedia Commons for tiny smoke tests

**Verification**
Run the test suite with:

```bash
python -m pytest -q
```
