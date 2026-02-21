# Quantifying Dimensional Independence in Speech

Reproducibility code for the INTERSPEECH 2026 submission.

## What this repo currently does

- Estimates MI with MINE/CLUB/KSG from cached features
- Generates selected publication figures/tables from cache
- Writes standard outputs to `Results/`
- Writes improved visuals to `Results1/`

## Repository structure

```
P1/
├── src/
│   ├── run_experiment.py
│   ├── generate_results.py
│   ├── improved_visuals.py
│   └── prepare_data.py
├── Data/
├── output/
├── Results/
├── Results1/
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Put datasets under `Data/` using folder names used by scripts:

- `RAVDESS`
- `IEMOCAP`
- `L2-ARCTIC`
- `GMU-Accented Speech Archive`
- `UA-Speech`
- `MDVR-KCL`

### Dataset links

| Dataset | Source | Access |
|---|---|---|
| RAVDESS | [Zenodo](https://zenodo.org/record/1188976) | Open |
| IEMOCAP | [USC SAIL](https://sail.usc.edu/iemocap/) | Request required |
| L2-ARCTIC | [Psi Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/) | Open |
| GMU-Accented Speech Archive | [accent.gmu.edu](https://accent.gmu.edu/) | Open |
| UA-Speech | [UIC](http://www.isle.illinois.edu/sst/data/UASpeech/) | Request required |
| MDVR-KCL | [Zenodo](https://zenodo.org/record/2867216) | Open |

Optional deterministic file-list generation:

```bash
python src/prepare_data.py --data-dir Data --max-samples 500
```

## Configuration and hyperparameters

Primary configuration is in `configs/experiment_config.yaml` (where present) and script defaults in `src/generate_results.py` / `src/run_experiment.py`.

Common run settings:

| Parameter | Value |
|---|---|
| Global seed | 42 |
| Ensemble size | 3 |
| Max epochs | 100 |
| Batch size | 256 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Hidden dim | 256 |
| KSG k | 5 |
| EMA alpha | 0.01 |
| Log-var clamp | [-6, 2] |
| Gradient clip | 1.0 |
| Early-stop gap | 0.1 |
| Samples per dataset | 500 |

MI final estimate used in this codebase:

`I_final = (1 - w) * (I_MINE + I_CLUB) / 2 + w * I_KSG`,
with `w = 0.3` if `Δ <= 1.0`, else `w = min(0.6, 0.3 + 0.1 * Δ)`.

## Run

### 1) (Optional) build a fresh cache from raw audio

```bash
python src/run_experiment.py \
    --ravdess Data/RAVDESS \
    --iemocap Data/IEMOCAP \
    --l2arctic Data/L2-ARCTIC \
    --gmu "Data/GMU-Accented Speech Archive" \
    --uaspeech Data/UA-Speech \
    --mdvr Data/MDVR-KCL \
    --output-dir output \
    --max-samples 500 \
    --cache-dir output/cache_new
```

### 2) Generate results from cache

```bash
python src/generate_results.py --cache-dir output/cache
```

Use `output/cache_new` instead if you generated a new cache.

## Current outputs (important)

### `Results/` (from `generate_results.py`)

Generated figures:

- `P1_Extra1` (PDF/PNG)
- `P1_fig3_convergence` (PDF/PNG)

Generated tables:

- `P1_table1_features` (TEX)
- `P1_table2_mi_summary` (CSV)
- `P1_table2_mi_mean_sd` (CSV/TEX)
- `P1_table2_mi` (TEX)
- `P1_table3_attribution` (CSV/TEX)
- `P1_all_tables` (TEX)

Also generated:

- `results_v5.json`

Not generated (currently commented out):

- `P1_fig3_attribution`
- `P1_fig_mi_summary`
- `P1_fig_uncertainty`
- `P1_fig_convergence_2`
- `P1_table2_mi_summary_2` (CSV/TEX)

### `Results1/` (from `improved_visuals.py` via `generate_results.py`)

Generated figures:

- `P1_fig2_mi_heatmap` (PDF/PNG)
- `P1_fig4_attribution` (PDF/PNG)

Not generated (currently commented out):

- all `Results1/Tables` exports

## Notes

- `run_experiment.py` is used to build cache artifacts.
- `generate_results.py` is the main script for final figures/tables.
- Seeds are fixed (`SEED = 42`) in the scripts for repeatability.

## License

Code is provided for research reproducibility. Dataset usage follows each dataset’s own license/terms.
