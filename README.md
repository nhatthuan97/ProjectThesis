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
| `Code/` | Experiments. `01_baseline_fullscale`: centralized multi-model baseline + five FL aggregation methods (FedAvg/FedProx/FedAvgM/FedAdam/SCAFFOLD). `02_consent_churn`: the three-regime consent-churn study. See `Code/README.md` for environment setup. |
| `Disertation_at_Work/` | Dissertation proposal (`main.tex`) and figures from the published component papers (Spark benchmarking, blockchain/IPFS consent layer, COVID supply-chain study). |

Grant drafts, meeting slides, and personal progress logs are kept locally and
not published in this repository.

## Status

Utility-cost axis (RQ2) executed and H2 confirmed. Next: MLP amplification
check, then the systems-cost axis (on-chain gas/latency instrumentation driven
by the same churn schedules), then the joint frontier.
