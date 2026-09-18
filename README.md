# The Cost of Patient Autonomy in Cross-Silo Federated Learning

PhD dissertation project — measuring what it actually *costs* to give patients
fine-grained, dynamic, revocable consent over their data in federated healthcare
machine learning.

## The idea

Hospitals want to train clinical models together without sharing patient
records (federated learning), and a growing literature adds a blockchain layer
so patients can grant or revoke consent to participate at any time. The field
treats that capability as an unqualified good — but nobody has measured its
price. This project quantifies the two-axis **cost of patient autonomy**:

- **Systems cost** — on-chain transaction, latency, and bandwidth overhead that
  consent governance adds to every training round.
- **Utility cost** — model accuracy lost because revocable consent makes the
  training population non-stationary (the non-IID regime where federated
  averaging degrades).

The headline deliverable is a joint **cost–utility frontier**: how both costs
move as consent churn, refresh cadence, federation size, and chain backend are
swept — and the breaking points where patient-controlled federated learning
stops being practical.

## Key result so far

On the primary task (30-day readmission, UCI Diabetes 130-US-hospitals,
~102k real encounters, 3 silos, 5 seeds): **who leaves matters more than how
many leave.** Removing one positive-heavy silo costs −0.059 AUROC while
removing the *same number* of patients at random costs −0.039; transient and
random per-patient churn are nearly free (−0.003 to −0.007). The damage comes
from distribution shift, not headcount.

## Repository layout

| Folder | Contents |
|---|---|
| `Code/` | Experiments, and only experiments. `01_baseline_fullscale`: centralized multi-model baseline + five FL aggregation methods (FedAvg/FedProx/FedAvgM/FedAdam/SCAFFOLD). `02_consent_churn`: the three-regime consent-churn study. `experiment_setup.py` holds the one definition of the data split and the silo partition. See `Code/README.md` for environment setup. |
| `results/` | Everything generated: the metric JSONs the experiments write, and `figures/` built from them by `Proposal_Defense/make_figs.py`. Never edited by hand. |
| `assets/` | Static figures from the published component papers (Spark benchmarking, blockchain/IPFS consent layer, COVID supply-chain study). Inputs, not outputs. |
| `Disertation_at_Work/` | Dissertation proposal (`main.tex`) and its bibliography. |

Every LaTeX document pulls figures from `results/figures/` and `assets/` via
`\graphicspath`, so there is exactly one copy of each figure and rerunning an
experiment propagates to every document.

## Status

Utility-cost axis (RQ2) executed and H2 confirmed, including the MLP
amplification check: rerunning the full churn grid with a higher-capacity MLP
client reproduces the logreg deltas within noise, so the churn-cost structure
is model-independent. Next: the systems-cost axis (on-chain gas/latency
instrumentation driven by the same churn schedules), then the joint frontier.
