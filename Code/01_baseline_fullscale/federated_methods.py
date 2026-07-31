"""Federated averaging methods for an averageable model (logistic regression).

Implements five server-side aggregation strategies on the same LogReg client so
they are compared apples-to-apples:

  FedAvg    -- weighted parameter average                (McMahan 2017)
  FedProx   -- FedAvg + client proximal term mu          (Li 2020)
  FedAvgM   -- server momentum on the aggregated delta   (Hsu 2019)
  FedAdam   -- adaptive server optimizer (FedOpt family) (Reddi 2021)
  SCAFFOLD  -- client control variates correct drift     (Karimireddy 2020)

All support class-weighting for the imbalance. Trees (RF/XGB) are NOT averageable,
so the federated ceiling is set by the best averageable model, not the best model.
"""
from __future__ import annotations
import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _class_weights(y, balanced):
    if not balanced:
        return None
    pos = y.mean()
    return (0.5 / pos, 0.5 / (1 - pos))          # (w_pos, w_neg), global


def _local_update(theta0, X, y, *, epochs, lr, l2, bs, rng, cw,
                  prox_mu=0.0, c_corr=None):
    """One client's local SGD from theta0. Returns (theta, n_steps).

    prox_mu : FedProx proximal pull toward theta0.
    c_corr  : SCAFFOLD correction vector (c_global - c_client), added to grad.
    """
    theta = theta0.copy()
    n = len(y)
    sw = np.ones(n) if cw is None else np.where(y == 1, cw[0], cw[1])
    steps = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            bi = order[s:s + bs]
            Xb, yb, sb = X[bi], y[bi], sw[bi]
            w, b = theta[:-1], theta[-1]
            err = (sigmoid(Xb @ w + b) - yb) * sb
            den = sb.sum()
            grad = np.concatenate([Xb.T @ err / den + l2 * w, [err.sum() / den]])
            if prox_mu:
                grad = grad + prox_mu * (theta - theta0)
            if c_corr is not None:
                grad = grad + c_corr
            theta = theta - lr * grad
            steps += 1
    return theta, steps


def federated_train(X, y, silos, Xte, *, method="fedavg", nf,
                    rounds=40, local_epochs=1, lr=0.1, l2=1e-4, bs=256,
                    balanced=True, seed=0, round_silos=None,
                    prox_mu=0.1, server_momentum=0.9,
                    server_lr=0.05, adam_b1=0.9, adam_b2=0.99, adam_eps=1e-3):
    """Train with the chosen aggregation method; return test probabilities.

    round_silos : optional callable r -> list of K active-index arrays for round r
                  (consent churn). If None, the static `silos` participate every
                  round. Silos with 0 active patients this round contribute nothing;
                  aggregation weights are recomputed each round from active sizes.
    """
    cw = _class_weights(y, balanced)
    theta = np.zeros(nf + 1)

    # method state
    v = np.zeros(nf + 1)                 # FedAvgM velocity
    m_a = np.zeros(nf + 1); v_a = np.zeros(nf + 1)          # FedAdam moments
    c_global = np.zeros(nf + 1)
    c_clients = [np.zeros(nf + 1) for _ in silos]           # SCAFFOLD

    for r in range(rounds):
        cur = round_silos(r) if round_silos is not None else silos
        sizes_r = np.array([len(s) for s in cur], float)
        if sizes_r.sum() == 0:            # nobody consents this round -> skip
            continue
        wts = sizes_r / sizes_r.sum()
        deltas = []                       # client model deltas (theta_k - theta)
        dc = []                           # SCAFFOLD control-variate deltas
        for sid, idx in enumerate(cur):
            if len(idx) == 0:
                deltas.append(np.zeros(nf + 1)); dc.append(np.zeros(nf + 1)); continue
            rng = np.random.default_rng(seed + 1000 * r + sid)
            c_corr = (c_global - c_clients[sid]) if method == "scaffold" else None
            th_k, steps = _local_update(
                theta, X[idx], y[idx], epochs=local_epochs, lr=lr, l2=l2, bs=bs,
                rng=rng, cw=cw, prox_mu=(prox_mu if method == "fedprox" else 0.0),
                c_corr=c_corr)
            deltas.append(th_k - theta)
            if method == "scaffold":
                # Option II: c_i+ = c_i - c + (x - y_i)/(K * lr)
                new_ci = c_clients[sid] - c_global + (theta - th_k) / (steps * lr)
                dc.append(new_ci - c_clients[sid])
                c_clients[sid] = new_ci

        avg_d = np.average(np.stack(deltas), axis=0, weights=wts)

        if method in ("fedavg", "fedprox"):
            theta = theta + avg_d
        elif method == "fedavgm":
            v = server_momentum * v + avg_d
            theta = theta + v
        elif method == "fedadam":
            m_a = adam_b1 * m_a + (1 - adam_b1) * avg_d
            v_a = adam_b2 * v_a + (1 - adam_b2) * (avg_d ** 2)
            theta = theta + server_lr * m_a / (np.sqrt(v_a) + adam_eps)
        elif method == "scaffold":
            theta = theta + avg_d           # server_lr_g = 1
            c_global = c_global + np.average(np.stack(dc), axis=0,
                                             weights=np.ones(len(silos)))
        else:
            raise ValueError(method)

    return sigmoid(Xte @ theta[:-1] + theta[-1])


METHODS = ["fedavg", "fedprox", "fedavgm", "fedadam", "scaffold"]


if __name__ == "__main__":
    # standalone sanity check: IID should ~match centralized; skew should hurt
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from best_single_baseline import load_xy, NUMERIC

    X, y, feat = load_xy()
    num_idx = [i for i, n in enumerate(feat) if n in NUMERIC]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    mu = Xtr[:, num_idx].mean(0); sd = Xtr[:, num_idx].std(0) + 1e-8
    Xtr = Xtr.copy(); Xte = Xte.copy()
    Xtr[:, num_idx] = (Xtr[:, num_idx] - mu) / sd
    Xte[:, num_idx] = (Xte[:, num_idx] - mu) / sd
    nf = X.shape[1]

    def partition(y, k, alpha, seed=0):
        rng = np.random.default_rng(seed); silos = [[] for _ in range(k)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0]); pr = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(pr) * len(idx)).astype(int)[:-1]
            for s, ch in enumerate(np.split(idx, cuts)): silos[s].extend(ch.tolist())
        return [np.sort(np.array(s, int)) for s in silos]

    for alpha in (10.0, 0.5, 0.1):
        silos = partition(ytr, 3, alpha, 0)
        print(f"\nalpha={alpha}")
        for meth in METHODS:
            p = federated_train(Xtr, ytr, silos, Xte, method=meth, nf=nf, seed=0)
            print(f"  {meth:<10s} AUROC={roc_auc_score(yte, p):.4f}")
