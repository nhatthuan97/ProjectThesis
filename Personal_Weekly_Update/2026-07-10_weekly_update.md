# Weekly Update — Week ending 2026-07-10

**Project:** The Cost of Patient Autonomy in Cross-Silo Federated Learning
**Focus this week:** Locking the dataset decision and standing up the two measurement
instruments (federated baseline + the utility-cost / consent-churn axis) end to end.

---

## TL;DR

- **Pivoted the data strategy.** Dropped the cervical-cancer dataset (too small to power a
  churn study) and the Synthea-for-accuracy plan (weak signal pollutes the utility
  measurement). Moved to **UCI Diabetes 130-US-hospitals** (~102k real encounters), which
  does the job of *both* the old sources at once — real signal *and* enough scale to power
  the degradation study.
- **Built and executed two code parts** (baseline + federated methods; consent-churn study),
  both reproducible with embedded results.
- **Confirmed the central hypothesis (H2): "who leaves" matters more than "how many leave."**
- **Clarified the role of each dataset and where the thesis novelty actually lives** (the
  joint frontier + the churn finding, *not* the blockchain build).

---

## Decisions made

### 1. Dataset pivot (cervical + Synthea → Diabetes 130)
- **Cervical (858 rows) is too small.** The binding number is *54 positive cases*. Across 3
  silos with up to 70% churn, positives drop to single digits per silo — statistically dead
  for measuring degradation deltas.
- **Synthea is the wrong tool for the utility axis.** Its signal is weak by construction, so
  accuracy-degradation curves computed on it would measure generator artifacts, not real
  utility loss. It also cannot be "joined" to a real dataset (no shared population/keys).
- **Diabetes 130 collapses the old two-source workaround into one better source:** large
  (100k rows, ~11k positives) *and* real signal. Task = 30-day readmission (binary, 11%
  positive).

### 2. Synthea's real role = the systems axis, not accuracy
- On the systems (traffic/compute) axis, **accuracy is irrelevant** — only transaction volume
  and payload size matter. That is where Synthea earns its keep: generating FHIR-native
  patient records at hospital scale to stress-test governance cost.
- Key realization: **the consent-churn schedules already built ARE the traffic generator** —
  every withdrawal/restore = one on-chain transaction. So a single churn parameter drives
  *both* axes (utility and systems) simultaneously. That coupling is what makes a joint
  cost–utility frontier coherent.

### 3. Where the novelty lives (framing correction)
- The inability to join two datasets is **not** the research gap — it's a plumbing fact. The
  gap is the *unmeasured two-axis cost of patient-controlled consent in FL*.
- The blockchain/IPFS system is the **instrument**, not the contribution (it was already
  published). Novelty = the **joint cost–utility frontier**, the **three-regime "who > how
  many" finding**, and the **on-chain infeasibility threshold**.
- Strong version of the plan: build the **joint frontier on ONE population (Diabetes)** so
  both axes actually meet; use **Synthea only to scale the systems axis** and find the
  feasibility threshold. Avoid the weak version where the two datasets are disconnected halves.

---

## What was built (in `Code/`)

- Shared `data/` root + numbered per-part folders; conda env `thesis` (Python 3.11), Jupyter
  kernel "Python (thesis)".
- **`01_baseline_fullscale/`**
  - `best_single_baseline.py` — multi-model centralized benchmark (LogReg / RF / HGB /
    XGBoost / LightGBM), imbalance-aware, 5-fold CV.
  - `federated_methods.py` — five aggregation methods (FedAvg, FedProx, FedAvgM, FedAdam,
    SCAFFOLD) on a transparent logistic-regression client; now churn-aware.
  - `baseline_and_federated_methods.ipynb` — executed comparison.
- **`02_consent_churn/`**
  - `churn.py` — churn schedules (transient / permanent-random / permanent-biased /
    whole-silo / count-matched control).
  - `consent_churn_study.ipynb` — executed 5-seed study.

---

## Key results

### Centralized baseline (the ceiling)
- Best model (RandomForest / XGBoost): **AUROC ≈ 0.677** — sits right in the published SOTA
  band (0.667–0.70). Model choice barely moves AUROC; the task signal ceiling is genuinely low.
- Best *averageable* model (logistic regression): 0.666. Trees are not weight-averageable, so
  the **federated ceiling is ~0.011 below the tree ceiling** — the concrete cost of federating.

### Federated averaging methods (K = 3)
- AUROC and PR-AUC are **essentially flat across all five methods** at IID and moderate skew
  (~0.666–0.670) — at K=3 the aggregation choice barely matters until data is pathological.
- **SCAFFOLD collapses at severe skew** (α=0.1: AUROC 0.625) — control variates are unstable
  with few, tiny, extreme silos. A documented small-K failure mode.
- Practical default: **FedAvg (class-weighted)** — robust and already at the ceiling.

### Consent-churn study (the utility-cost axis) — Δ AUROC vs no churn, 5 seeds
| Regime | Effect |
|---|---|
| Transient (rejoin), 70% | −0.003 (≈ free) |
| Permanent random, 70% | −0.007 (≈ free) |
| **Permanent biased (positives leave first), 70%** | **−0.021** (~3× — distribution shift) |
| Whole-silo, remove positive-light silo (α=0.1) | −0.010 |
| **Whole-silo, remove positive-heavy silo (α=0.1)** | **−0.059** (~6×) |

- **Isolation test (same patient count removed):** whole-silo departure −0.059 vs count-matched
  random −0.039 → **who leaves > how many leave. H2 confirmed.**
- The three regimes are demonstrably different — the separation the literature conflates.

---

## Honest caveats / risks

- **Magnitudes are modest** because the dataset is large, logistic regression saturates, and
  AUROC is rank-robust. Random per-patient churn is genuinely *cheap* here (a real but
  undramatic finding). The large costs come from **distribution shift** (biased withdrawal,
  whole-silo departure), not headcount.
- To make the utility axis larger and the frontier sharper, the levers are: a higher-capacity
  averageable model (**MLP**), higher heterogeneity, targeted/biased withdrawal, and ultimately
  **real-hospital silos (eICU)**.
- Switching to Diabetes shifts the disease narrative from cervical screening to readmission.
  Methodology is disease-agnostic, but the grant/dissertation text is written around cervical
  and will need updating. Suggested reconciliation: Diabetes = powered primary; cervical =
  small secondary "signal-validity anchor" + continuity with prior published work.

---

## Next steps (priority order)

1. **MLP amplification check** — swap the logistic-regression client for a small MLP to test
   whether the small churn magnitudes are a linear-model artifact. Quick, high value.
2. **Systems-cost axis** — wire on-chain consent measurement (gas / latency / bandwidth per
   round) onto the existing per-round hook, driven by the churn schedules as the traffic
   generator.
3. **Joint cost–utility frontier** — same population (Diabetes), both axes vs the shared churn
   knob. This is the central scientific artifact.
4. **Synthea scaling study** — push population to hospital scale on the systems axis to locate
   the per-round on-chain consent infeasibility threshold (RQ1).
5. **Multi-project consent matrix** (RQ4).
