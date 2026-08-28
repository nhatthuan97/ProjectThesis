# Thesis code

Measuring the cost of patient autonomy in consent-gated cross-silo federated
learning. Each numbered folder is one self-contained part of the work; they all
link to the **shared `data/` folder** as the single source of raw data.

```
Code/
  data/                          # shared raw data (git-ignore the big files)
    diabetes_raw.csv             # UCI Diabetes 130-US-hospitals (auto-downloaded)
  01_baseline_fullscale/
    baseline_full_scale_accuracy.ipynb       # exploratory: ceiling/floor/FedAvg + non-IID sweeps
    baseline_and_federated_methods.ipynb     # FULL: multi-model baseline + 5 FL methods @ K=3
    best_single_baseline.py                  # module: preprocessing + multi-model CV benchmark
    federated_methods.py                     # module: FL methods, logreg + MLP flat-vector clients
    best_single_baseline_results.json        # saved centralized model ranking
  02_consent_churn/
    churn.py                                 # module: consent-churn schedules + runner
    consent_churn_study.ipynb                # FULL: 3-regime churn study, 5 seeds, H2 test
    mlp_amplification_study.py               # runner: full churn grid on logreg AND MLP clients
    mlp_amplification.ipynb                  # results + findings of the amplification check
    mlp_amplification_results.json           # saved per-run metrics (200 runs)
  ...                            # (later) 03_blockchain_consent/ ...
```

## Environment

```bash
conda create -n thesis python=3.11 -y
conda run -n thesis pip install numpy pandas scikit-learn scipy matplotlib \
    ucimlrepo jupyterlab nbformat nbconvert ipykernel
conda run -n thesis python -m ipykernel install --user --name thesis \
    --display-name "Python (thesis)"
```

## Data convention

Notebooks reference the shared data with a relative path, e.g. from a numbered
folder: `Path("../data")`. The baseline notebook **auto-downloads** the dataset
to `../data/` on first run, so any sibling folder can rely on it being there.
Keep raw data out of version control (see `.gitignore`).

## Parts

| Folder | What it does | Status |
|---|---|---|
| `01_baseline_fullscale` | Multi-model centralized baseline (at published SOTA, AUROC ~0.677) + five FL averaging methods (FedAvg/FedProx/FedAvgM/FedAdam/SCAFFOLD) compared across 3 silos and IID→severe non-IID. Establishes the ceiling/floor/federated reference anchors. No consent churn yet. | done |
| `02_consent_churn` | Inject transient / permanent(random) / permanent(biased) / whole-silo consent churn into the FedAvg round loop; measure utility degradation vs baselines over 5 seeds. Confirms H2 (who leaves > how many leave) via a count-matched isolation test. `churn.py` = schedules; `consent_churn_study.ipynb` = executed study. | done |
| `02_consent_churn` (MLP amplification check) | Rerun the full churn grid with a higher-capacity averageable client (1-hidden-layer MLP, 64 units, centralized AUROC 0.670 vs logreg 0.666) on identical partitions/schedules. Result: **no amplification** — deltas match logreg within noise, so the churn-cost structure is model-independent and H2 survives the model swap. | done |
