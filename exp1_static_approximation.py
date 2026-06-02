#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
static_approximation_two_exps.py

Two experiments for validating a *static* score approximation on an isotropic GMM under an OU flow.

Experiment A (sigma sweep):
    - vary sigma0 in {0.01, 0.1, 1.0}
    - fixed width, fixed depth
    - report RelMSE(t) at t/T in {0.1,0.3,0.5,0.7,0.9}
    - averaged over N seeds, report mean and std

Experiment B (width sweep):
    - vary width in {128,256,512,1024,2048}
    - fixed sigma0, fixed depth
    - report RelMSE(t) at t/T in {0.1,0.3,0.5,0.7,0.9}
    - averaged over N seeds, report mean and std

Implementation notes
- Closed-form score for isotropic GMM marginal p_t:
    score(x,t) = ∇_x log p_t(x) = -(1/s_t^2) (x - \bar m(x,t)),
    with responsibilities computed stably and memory-efficiently.
- Avoids (N,K,d) allocations by using ||x-m||^2 = ||x||^2 + ||m||^2 - 2 x m^T.
- Uses tqdm progress on the full plan.
- Optional caching per run: --resume.

Outputs
- per_run_results.csv                (all runs)
- exp_sigma_agg_long.csv             (long-form aggregated)
- exp_sigma_agg_pivot.csv            (wide/pivot aggregated, easier to read)
- exp_width_agg_long.csv
- exp_width_agg_pivot.csv

No plotting by default. (You can keep your visualization code separately.)
"""

from __future__ import annotations

import os
import math
import csv
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    return m, sd


def fmt_sci2(x: float) -> str:
    # 2 significant digits in mantissa (rounded), scientific notation like 6.56e-3
    return f"{x:.2e}"


def fmt_mean_std_sci2(m: float, sd: float) -> str:
    return f"{fmt_sci2(m)} ± {fmt_sci2(sd)}"


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def pivot_table(
    rows: List[Dict[str, Any]],
    row_key: str,
    col_key: str,
    val_key: str,
    row_order: List[Any],
    col_order: List[Any],
) -> List[Dict[str, Any]]:
    """
    Build a "wide" table:
        each output row corresponds to a unique row_key value,
        columns correspond to col_key values,
        cell is rows[(row_key, col_key)][val_key] if exists else "".
    """
    M: Dict[Tuple[Any, Any], Any] = {}
    for r in rows:
        M[(r[row_key], r[col_key])] = r[val_key]
    out: List[Dict[str, Any]] = []
    for rk in row_order:
        rr: Dict[str, Any] = {row_key: rk}
        for ck in col_order:
            rr[str(ck)] = M.get((rk, ck), "")
        out.append(rr)
    return out


# -----------------------------
# Model: alternating ReLU / ReQU
# -----------------------------

class ReQU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)


def build_mlp(d: int, width: int, depth: int, act0: str = "relu") -> nn.Module:
    """
    depth = number of hidden layers
    activation alternates: ReLU, ReQU, ReLU, ReQU, ...
    """
    assert depth >= 1
    acts: List[nn.Module] = []
    for i in range(depth):
        if (i % 2 == 0 and act0 == "relu") or (i % 2 == 1 and act0 == "requ"):
            acts.append(nn.ReLU())
        else:
            acts.append(ReQU())

    layers: List[nn.Module] = []
    in_dim = d
    for i in range(depth):
        layers.append(nn.Linear(in_dim, width))
        layers.append(acts[i])
        in_dim = width
    layers.append(nn.Linear(in_dim, d))

    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


# -----------------------------
# GMM-OU: sampling + closed-form score (memory-safe)
# -----------------------------

@dataclass
class GMMOU:
    d: int
    K: int
    mu: torch.Tensor          # (K,d)
    pi: torch.Tensor          # (K,)
    sigma0: float
    beta: float
    sigma_inf: float = 1.0

    def gmm_ou_params(self, t: float) -> Tuple[torch.Tensor, float]:
        alpha = math.exp(-self.beta * t)
        m_t = self.mu * alpha
        s2_t = (alpha * alpha) * (self.sigma0 ** 2) + (1.0 - alpha * alpha) * (self.sigma_inf ** 2)
        return m_t, float(s2_t)

    @torch.no_grad()
    def sample_xt(self, t: float, n: int, device: torch.device) -> torch.Tensor:
        m_t, s2_t = self.gmm_ou_params(t)
        m_t = m_t.to(device)
        pi = self.pi.to(device)

        idx = torch.multinomial(pi, num_samples=n, replacement=True)
        means = m_t[idx]
        eps = torch.randn(n, self.d, device=device)
        return means + math.sqrt(s2_t) * eps

    @torch.no_grad()
    def _dist2_x_m(self, x: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        """
        x: (N,d), m_t: (K,d) -> dist2: (N,K), without allocating (N,K,d).
        """
        x = x.float()
        m_t = m_t.float()
        x2 = (x * x).sum(dim=-1, keepdim=True)          # (N,1)
        m2 = (m_t * m_t).sum(dim=-1).view(1, -1)        # (1,K)
        xm = x @ m_t.t()                                 # (N,K)
        return (x2 + m2 - 2.0 * xm).clamp_min(0.0)

    @torch.no_grad()
    def responsibilities(self, x: torch.Tensor, t: float) -> torch.Tensor:
        m_t, s2_t = self.gmm_ou_params(t)
        m_t = m_t.to(x.device).float()
        pi = self.pi.to(x.device).float().clamp_min(1e-30)
        log_pi = torch.log(pi).view(1, -1)              # (1,K)

        dist2 = self._dist2_x_m(x, m_t)                 # (N,K)
        logits = log_pi - 0.5 * dist2 / float(s2_t)
        return torch.softmax(logits, dim=-1)            # (N,K)

    @torch.no_grad()
    def score_closed_form(self, x: torch.Tensor, t: float, chunk: int = 0) -> torch.Tensor:
        """
        score(x,t) = -(x - \bar m(x,t))/s_t^2, where \bar m(x,t)=Σ_k γ_k m_k(t).
        chunk>0 computes in batches to reduce peak memory for large N.
        """
        m_t, s2_t = self.gmm_ou_params(t)
        m_t = m_t.to(x.device).float()

        if chunk is None or chunk <= 0:
            gamma = self.responsibilities(x, t)         # (N,K)
            bar_m = gamma @ m_t                         # (N,d)
            return -(x.float() - bar_m) / float(s2_t)

        outs = []
        N = x.shape[0]
        for i in range(0, N, chunk):
            xb = x[i:i + chunk]
            gamma = self.responsibilities(xb, t)
            bar_m = gamma @ m_t
            outs.append(-(xb.float() - bar_m) / float(s2_t))
        return torch.cat(outs, dim=0)


# -----------------------------
# Training and evaluation
# -----------------------------

@dataclass
class TrainCfg:
    device: str = "cuda"
    lr: float = 2e-4
    weight_decay: float = 0.0
    steps: int = 20000
    batch_size: int = 1024
    grad_clip: float = 1.0
    eval_n: int = 50000
    eval_batch: int = 1024
    score_chunk: int = 0
    print_every: int = 500


@torch.no_grad()
def estimate_relmse(model: nn.Module, gmm: GMMOU, t: float, cfg: TrainCfg, device: torch.device) -> float:
    model.eval()
    total_num = 0.0
    total_den = 0.0
    seen = 0

    while seen < cfg.eval_n:
        n = min(cfg.eval_batch, cfg.eval_n - seen)
        x = gmm.sample_xt(t=t, n=n, device=device)
        y = gmm.score_closed_form(x, t=t, chunk=cfg.score_chunk)
        yhat = model(x).float()

        total_num += float((yhat - y).pow(2).sum(dim=-1).sum().detach().cpu())
        total_den += float((y).pow(2).sum(dim=-1).sum().detach().cpu())
        seen += n

    return total_num / max(total_den, 1e-12)


def train_for_t(
    gmm: GMMOU,
    t: float,
    cfg: TrainCfg,
    width: int,
    depth: int,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_mlp(d=gmm.d, width=width, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        x = gmm.sample_xt(t=t, n=cfg.batch_size, device=device)
        y = gmm.score_closed_form(x, t=t, chunk=cfg.score_chunk)
        yhat = model(x).float()
        loss = F.mse_loss(yhat, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if verbose and (step % cfg.print_every == 0 or step == 1):
            with torch.no_grad():
                num = (yhat - y).pow(2).sum(dim=-1).mean()
                den = y.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
                rel = float((num / den).detach().cpu())
            print(f"[t={t:.4g}] step {step:6d}/{cfg.steps}  loss={loss.item():.4e}  relmse(batch)={rel:.4e}")

    train_time = time.time() - t0
    eval_rel = estimate_relmse(model, gmm, t=t, cfg=cfg, device=device)

    metrics = {
        "t": float(t),
        "width": float(width),
        "depth": float(depth),
        "steps": float(cfg.steps),
        "relmse": float(eval_rel),
        "train_sec": float(train_time),
        "device_is_cuda": 1.0 if device.type == "cuda" else 0.0,
    }
    return model, metrics


# -----------------------------
# Caching per-run metrics
# -----------------------------

def run_id(
    exp_name: str,
    seed: int,
    sigma0: float,
    depth: int,
    width: int,
    t: float,
    steps: int,
    batch: int,
    lr: float,
    beta: float,
    mu_box: float,
    d: int,
    K: int,
) -> str:
    # stable & filesystem-friendly
    return (
        f"{exp_name}"
        f"_seed{seed}"
        f"_sig{sigma0:g}"
        f"_dep{depth}"
        f"_wid{width}"
        f"_t{t:.6f}"
        f"_steps{steps}"
        f"_bs{batch}"
        f"_lr{lr:g}"
        f"_beta{beta:g}"
        f"_box{mu_box:g}"
        f"_d{d}_K{K}"
    )


def maybe_load_metric(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
        if len(rows) != 1:
            return None
        return {k: float(v) for k, v in rows[0].items()}
    except Exception:
        return None


def save_metric(path: str, metric: Dict[str, float]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric.keys()))
        w.writeheader()
        w.writerow(metric)


# -----------------------------
# Experiment drivers
# -----------------------------

def run_experiment(
    exp_name: str,
    *,
    seeds: List[int],
    t_fracs: List[float],
    T: float,
    d: int,
    K: int,
    mu_box: float,
    beta: float,
    sigma_inf: float,
    sigmas: List[float],
    widths: List[int],
    depth: int,
    cfg: TrainCfg,
    out_dir: str,
    resume: bool,
    deterministic: bool,
    vary: str,  # "sigma" or "width"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
        long_rows: aggregated long-form rows
        pivot_rows: pivoted wide-form rows
    """

    assert vary in ("sigma", "width")

    t_list = [float(T) * f for f in t_fracs]
    # pre-sample mu, pi per seed, shared across all conditions (sigma/width/t) for that seed
    mu_by_seed: Dict[int, torch.Tensor] = {}
    pi_by_seed: Dict[int, torch.Tensor] = {}

    cache_dir = os.path.join(out_dir, "cache_runs")
    ensure_dir(cache_dir)

    # per-run logging CSV (append)
    per_run_csv = os.path.join(out_dir, "per_run_results.csv")
    per_run_fields = [
        "exp", "seed",
        "sigma0", "depth", "width", "t", "t_frac",
        "relmse", "train_sec",
        "beta", "mu_box", "d", "K",
        "steps", "batch_size", "lr", "eval_n", "eval_batch", "score_chunk",
    ]
    if not os.path.isfile(per_run_csv):
        write_csv(per_run_csv, [], per_run_fields)

    # aggregation bucket
    # keys:
    #   sigma-exp: (sigma0, t_frac) -> [relmse over seeds]
    #   width-exp: (width, t_frac)  -> [relmse over seeds]
    bucket: Dict[Tuple[float, float], List[float]] = defaultdict(list)

    plan: List[Tuple[int, float, int, float, float]] = []
    # Each item: (seed, sigma0, width, t, t_frac)
    if vary == "sigma":
        for seed in seeds:
            for sigma0 in sigmas:
                for tf, t in zip(t_fracs, t_list):
                    plan.append((seed, float(sigma0), int(widths[0]), float(t), float(tf)))
    else:
        for seed in seeds:
            for width in widths:
                for tf, t in zip(t_fracs, t_list):
                    plan.append((seed, float(sigmas[0]), int(width), float(t), float(tf)))

    pbar = tqdm(plan, desc=f"{exp_name}", dynamic_ncols=True)
    for seed, sigma0, width, t, tf in pbar:
        if seed not in mu_by_seed:
            set_seed(seed, deterministic=deterministic)
            mu_by_seed[seed] = (torch.rand(K, d) * 2.0 - 1.0) * float(mu_box)
            pi_by_seed[seed] = torch.ones(K) / float(K)

        mu = mu_by_seed[seed]
        pi = pi_by_seed[seed]

        gmm = GMMOU(
            d=d, K=K,
            mu=mu, pi=pi,
            sigma0=float(sigma0),
            beta=float(beta),
            sigma_inf=float(sigma_inf),
        )

        rid = run_id(
            exp_name=exp_name,
            seed=seed,
            sigma0=sigma0,
            depth=depth,
            width=width,
            t=float(t),
            steps=cfg.steps,
            batch=cfg.batch_size,
            lr=cfg.lr,
            beta=beta,
            mu_box=mu_box,
            d=d,
            K=K,
        )
        cache_path = os.path.join(cache_dir, f"{rid}.csv")

        metric = maybe_load_metric(cache_path) if resume else None
        if metric is None:
            pbar.set_postfix({"seed": seed, "sig": f"{sigma0:g}", "wid": width, "t/T": f"{tf:g}"})
            _, metric0 = train_for_t(
                gmm=gmm, t=float(t), cfg=cfg,
                width=int(width), depth=int(depth),
                verbose=False,
            )
            metric = {
                "exp": 0.0,  # placeholder for numeric-only cache, overwritten in per_run CSV
                "seed": float(seed),
                "sigma0": float(sigma0),
                "depth": float(depth),
                "width": float(width),
                "t": float(t),
                "t_frac": float(tf),
                "relmse": float(metric0["relmse"]),
                "train_sec": float(metric0["train_sec"]),
                "beta": float(beta),
                "mu_box": float(mu_box),
                "d": float(d),
                "K": float(K),
                "steps": float(cfg.steps),
                "batch_size": float(cfg.batch_size),
                "lr": float(cfg.lr),
                "eval_n": float(cfg.eval_n),
                "eval_batch": float(cfg.eval_batch),
                "score_chunk": float(cfg.score_chunk),
            }
            save_metric(cache_path, metric)

            # append to per-run CSV with string exp
            row = dict(metric)
            row["exp"] = exp_name
            with open(per_run_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=per_run_fields)
                w.writerow(row)

        # bucket key
        if vary == "sigma":
            key = (float(sigma0), float(tf))
        else:
            key = (float(width), float(tf))
        bucket[key].append(float(metric["relmse"]))

    # Aggregate long rows
    long_rows: List[Dict[str, Any]] = []
    if vary == "sigma":
        for sigma0 in sigmas:
            for tf in t_fracs:
                xs = bucket[(float(sigma0), float(tf))]
                m, sd = mean_std(xs)
                long_rows.append(
                    {
                        "sigma0": f"{sigma0:g}",
                        "t/T": f"{tf:g}",
                        "mean": f"{m:.8e}",
                        "std": f"{sd:.8e}",
                        "mean±std": fmt_mean_std_sci2(m, sd),
                        "n_trials": str(len(xs)),
                        "depth": str(depth),
                        "width": str(widths[0]),
                    }
                )
        long_rows = sorted(long_rows, key=lambda r: (float(r["sigma0"]), float(r["t/T"])))
        row_order = [f"{s:g}" for s in sigmas]
        col_order = [f"{tf:g}" for tf in t_fracs]
        # pivot: rows = sigma0, cols = t/T, values = mean±std
        pivot_src = [{"sigma0": r["sigma0"], "t/T": r["t/T"], "cell": r["mean±std"]} for r in long_rows]
        pivot_rows = pivot_table(
            pivot_src,
            row_key="sigma0",
            col_key="t/T",
            val_key="cell",
            row_order=row_order,
            col_order=col_order,
        )
    else:
        for width in widths:
            for tf in t_fracs:
                xs = bucket[(float(width), float(tf))]
                m, sd = mean_std(xs)
                long_rows.append(
                    {
                        "width": str(int(width)),
                        "t/T": f"{tf:g}",
                        "mean": f"{m:.8e}",
                        "std": f"{sd:.8e}",
                        "mean±std": fmt_mean_std_sci2(m, sd),
                        "n_trials": str(len(xs)),
                        "depth": str(depth),
                        "sigma0": f"{sigmas[0]:g}",
                    }
                )
        long_rows = sorted(long_rows, key=lambda r: (int(r["width"]), float(r["t/T"])))
        row_order = [str(int(w)) for w in widths]
        col_order = [f"{tf:g}" for tf in t_fracs]
        pivot_src = [{"width": r["width"], "t/T": r["t/T"], "cell": r["mean±std"]} for r in long_rows]
        pivot_rows = pivot_table(
            pivot_src,
            row_key="width",
            col_key="t/T",
            val_key="cell",
            row_order=row_order,
            col_order=col_order,
        )

    return long_rows, pivot_rows


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="out_exp1_static_approximation")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")

    # GMM parameters
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--mu_box", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sigma_inf", type=float, default=1.0)

    # time grid
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--t_fracs", type=str, default="0.1,0.3,0.5,0.7,0.9")

    # training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_n", type=int, default=50000)
    parser.add_argument("--eval_batch", type=int, default=1024)
    parser.add_argument("--score_chunk", type=int, default=0)

    # shared architecture settings
    parser.add_argument("--depth", type=int, default=4)

    # Experiment A: sigma sweep
    parser.add_argument("--sigmas", type=str, default="0.01,0.1,1.0")
    parser.add_argument("--sigma_width", type=int, default=512)

    # Experiment B: width sweep
    parser.add_argument("--widths", type=str, default="128,256,512,1024,2048")
    parser.add_argument("--width_sigma", type=float, default=0.1)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "cache_runs"))

    seeds = parse_csv_ints(args.seeds)
    sigmas = parse_csv_floats(args.sigmas)
    widths = parse_csv_ints(args.widths)
    t_fracs = parse_csv_floats(args.t_fracs)

    cfg = TrainCfg(
        device=args.device,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        grad_clip=float(args.grad_clip),
        eval_n=int(args.eval_n),
        eval_batch=int(args.eval_batch),
        score_chunk=int(args.score_chunk),
    )

    # -------------------------
    # Experiment A: sigma sweep
    # -------------------------
    long_a, pivot_a = run_experiment(
        "exp_sigma",
        seeds=seeds,
        t_fracs=t_fracs,
        T=float(args.T),
        d=int(args.d),
        K=int(args.K),
        mu_box=float(args.mu_box),
        beta=float(args.beta),
        sigma_inf=float(args.sigma_inf),
        sigmas=sigmas,
        widths=[int(args.sigma_width)],   # fixed width
        depth=int(args.depth),
        cfg=cfg,
        out_dir=args.out_dir,
        resume=bool(args.resume),
        deterministic=bool(args.deterministic),
        vary="sigma",
    )

    write_csv(
        os.path.join(args.out_dir, "exp_sigma_agg_long.csv"),
        long_a,
        fieldnames=["sigma0", "t/T", "mean±std", "mean", "std", "n_trials", "depth", "width"],
    )
    # pivot: columns are t/T values (as strings)
    pivot_cols_a = ["sigma0"] + [f"{tf:g}" for tf in t_fracs]
    write_csv(
        os.path.join(args.out_dir, "exp_sigma_agg_pivot.csv"),
        pivot_a,
        fieldnames=pivot_cols_a,
    )

    # -------------------------
    # Experiment B: width sweep
    # -------------------------
    long_b, pivot_b = run_experiment(
        "exp_width",
        seeds=seeds,
        t_fracs=t_fracs,
        T=float(args.T),
        d=int(args.d),
        K=int(args.K),
        mu_box=float(args.mu_box),
        beta=float(args.beta),
        sigma_inf=float(args.sigma_inf),
        sigmas=[float(args.width_sigma)],  # fixed sigma0
        widths=widths,
        depth=int(args.depth),
        cfg=cfg,
        out_dir=args.out_dir,
        resume=bool(args.resume),
        deterministic=bool(args.deterministic),
        vary="width",
    )

    write_csv(
        os.path.join(args.out_dir, "exp_width_agg_long.csv"),
        long_b,
        fieldnames=["width", "t/T", "mean±std", "mean", "std", "n_trials", "depth", "sigma0"],
    )
    pivot_cols_b = ["width"] + [f"{tf:g}" for tf in t_fracs]
    write_csv(
        os.path.join(args.out_dir, "exp_width_agg_pivot.csv"),
        pivot_b,
        fieldnames=pivot_cols_b,
    )

    print("\nSaved outputs:")
    print(f"  {os.path.join(args.out_dir, 'exp_sigma_agg_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'exp_sigma_agg_pivot.csv')}")
    print(f"  {os.path.join(args.out_dir, 'exp_width_agg_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'exp_width_agg_pivot.csv')}")
    print(f"  {os.path.join(args.out_dir, 'per_run_results.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
