# Thesis code

Measuring the cost of patient autonomy in consent-gated cross-silo federated
learning. Each numbered folder is one self-contained part of the work; they all
link to the **shared `data/` folder** as the single source of raw data.

```
Code/
  data/                          # shared raw data (git-ignore the big files)
    diabetes_raw.csv             # UCI Diabetes 130-US-hospitals (auto-downloaded)
  experiment_setup.py            # shared: split/standardise, Dirichlet silo
                                 #   partition, and the results/ location
  01_baseline_fullscale/
    baseline_full_scale_accuracy.ipynb       # exploratory: ceiling/floor/FedAvg + non-IID sweeps
    baseline_and_federated_methods.ipynb     # FULL: multi-model baseline + 5 FL methods @ K=3
    best_single_baseline.py                  # module: preprocessing + multi-model CV benchmark
    federated_methods.py                     # module: FL methods, logreg + MLP flat-vector clients
  02_consent_churn/
    churn.py                                 # module: consent-churn schedules + runner
    consent_churn_study.ipynb                # FULL: 3-regime churn study, 5 seeds, H2 test
    mlp_amplification_study.py               # runner: full churn grid on logreg AND MLP clients
    mlp_amplification.ipynb                  # results + findings of the amplification check
  ...                            # (later) 03_blockchain_consent/ ...
```

Generated artifacts do NOT live here. Every script writes into `../results/`:

```
results/
  best_single_baseline_results.json   # centralized model ranking
  mlp_amplification_results.json      # 200 per-run metrics
  figures/                            # built by Proposal_Defense/make_figs.py
assets/                               # static figures from the published papers
```

Every LaTeX document reads `../results/figures/` and `../assets/` through its
`\graphicspath`, so rerunning an experiment and then `make_figs.py` updates the
dissertation, the proposal defense and the weekly deck with nothing copied by
hand. That is also why nothing under `results/` should ever be edited directly.


## Environment

Runs on itx-box in the `ds` virtualenv (`~/venvs/ds`), which also backs the
JupyterLab server and the "Python (ds)" kernel. `requirements.txt` pins the
versions the archived results were last verified against.

```bash
python3 -m venv ~/venvs/ds
~/venvs/ds/bin/pip install -r requirements.txt
~/venvs/ds/bin/python -m ipykernel install --user --name ds \
    --display-name "Python (ds)"
```

**Python 3.14 note.** On Linux, Python 3.14 changed the default multiprocessing
start method from `fork` to `forkserver`. `02_consent_churn/mlp_amplification_study.py`
shares the preloaded design matrix with its workers through a module-level
global, so it explicitly requests a `fork` context; under `forkserver` the
workers re-import the module and see an empty global. Keep that context if you
touch the runner.

### Verified reproduction (2026-09-18)

Rerun end to end on Python 3.14.7 / NumPy 2.5.3 / pandas 3.0.5 / scikit-learn
1.9.1 / XGBoost 3.4.1 / LightGBM 4.7.0:

| Artifact | Result |
|---|---|
| `mlp_amplification_results.json` (200 runs) | **byte-identical** to the archived copy |
| `federated_methods.py` | reproduces the flat method comparison and the SCAFFOLD collapse at alpha=0.1 (AUROC 0.6246) |
| `churn.py` | reproduces the regime ordering |
| `best_single_baseline_results.json` | reproduces except **XGBoost AUROC 0.676882 -> 0.676166** (-7.2e-04) under XGBoost 3.4.1 |

The XGBoost drift is a library-version effect, an order of magnitude below that
model's own CV standard deviation (+/-0.0064). It does not move the ranking, the
selected model, or the tuned threshold, all of which reproduce exactly. The
committed JSON is kept as the archived artifact rather than being overwritten.

## Data convention

Notebooks reference the shared data with a relative path, e.g. from a numbered
folder: `Path("../data")`. The baseline notebook **auto-downloads** the dataset
to `../data/` on first run, so any sibling folder can rely on it being there.
Keep raw data out of version control (see `.gitignore`).

## Parts

| Folder | What it does | Status |
|---|---|---|
| `01_baseline_fullscale` | Multi-model centralized baseline (AUROC 0.677, at the top of the published band for this dataset: 0.64-0.688 across Liu 2024 / Emi-Johnson 2025 / Salim 2026 -- see `consent-churn/references.bib`) + five FL averaging methods (FedAvg/FedProx/FedAvgM/FedAdam/SCAFFOLD) compared across 3 silos and IID→severe non-IID. Establishes the ceiling/floor/federated reference anchors. No consent churn yet. | done |
| `02_consent_churn` | Inject transient / permanent(random) / permanent(biased) / whole-silo consent churn into the FedAvg round loop; measure utility degradation vs baselines over 5 seeds. Confirms H2 (who leaves > how many leave) via a count-matched isolation test. `churn.py` = schedules; `consent_churn_study.ipynb` = executed study. | done |
| `02_consent_churn` (MLP amplification check) | Rerun the full churn grid with a higher-capacity averageable client (1-hidden-layer MLP, 64 units, centralized AUROC 0.670 vs logreg 0.666) on identical partitions/schedules. Result: **no amplification** — deltas match logreg within noise, so the churn-cost structure is model-independent and H2 survives the model swap. | done |
