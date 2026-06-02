#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exp_stability_balanced_vs_uniform.py

Stability-profile-balanced discretization vs uniform grids for one-step distillation

Goal
- Distill the full backward probability-flow transport Phi_{0<-T} by composing n one-step students
- Compare two grids for the same n
  (1) uniform time grid
  (2) stability-profile-balanced grid via A(t)=∫_0^t L(u) du and A(t_k)=k/n A(T)

Sweep
- n in {1,2,4,8}
- optional sigma0 sweep
- averaged over training seeds

Key alignment
- GMMOU data generation matches your earlier scripts
- Closed-form score matches your earlier scripts
- Student MLP uses ReLU-only hidden activations
- Teacher maps are computed by RK4 integrating the probability-flow ODE with closed-form score

Important design requirement implemented here
- Mixture means mu and weights pi are fixed across all runs (controlled only by --mu_seed)
- The trial seed controls only the training process (model init + minibatch sampling)
- Evaluation sampling is deterministic and independent of the trial seed, so seeds only measure training variance

Design choice for stability profile L(t)
- Uses a computable upper bound for the PF drift Lipschitz scale
    L(t) = beta * (abs|1 - 1/s_t^2| + diam(t)^2/(4 s_t^4))
  where s_t^2 is the OU marginal variance and diam(t)=max_{i,j}||m_i(t)-m_j(t)||

Outputs
- per_run_results.csv
- agg_long.csv
- agg_pivot_uniform.csv
- agg_pivot_stability.csv

No plotting.
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


def seed_mix(a: int, b: int, c: int = 0) -> int:
    """
    Simple deterministic seed mixer, returns a non-negative 32-bit integer.
    """
    x = (a & 0xFFFFFFFF) ^ ((b * 0x9E3779B1) & 0xFFFFFFFF) ^ ((c * 0x85EBCA6B) & 0xFFFFFFFF)
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return int(x)


# -----------------------------
# Model: ReLU MLP
# -----------------------------

def build_mlp(d: int, width: int, depth: int, act0: str = "relu") -> nn.Module:
    """
    depth = number of hidden layers
    This experiment uses ReLU-only hidden activations; act0 is retained for
    signature compatibility with the other experiment scripts.
    """
    _ = act0
    assert depth >= 1
    acts: List[nn.Module] = []
    for _ in range(depth):
        acts.append(nn.ReLU())

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
# GMM-OU: sampling + closed-form score
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
        log_pi = torch.log(pi).view(1, -1)

        dist2 = self._dist2_x_m(x, m_t)
        logits = log_pi - 0.5 * dist2 / float(s2_t)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def score_closed_form(self, x: torch.Tensor, t: float, chunk: int = 0) -> torch.Tensor:
        m_t, s2_t = self.gmm_ou_params(t)
        m_t = m_t.to(x.device).float()

        if chunk is None or chunk <= 0:
            gamma = self.responsibilities(x, t)
            bar_m = gamma @ m_t
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
# Teacher maps by RK4
# -----------------------------

@torch.no_grad()
def pf_velocity(x: torch.Tensor, t: float, gmm: GMMOU, score_chunk: int = 0) -> torch.Tensor:
    score = gmm.score_closed_form(x, t=t, chunk=score_chunk)
    return -gmm.beta * (x.float() + score)


@torch.no_grad()
def phi_t_to_s_rk4(
    x_t: torch.Tensor,
    t_from: float,
    t_to: float,
    gmm: GMMOU,
    n_steps: int,
    score_chunk: int = 0,
) -> torch.Tensor:
    """
    Integrate probability-flow ODE from t_from to t_to with RK4.
    Works for both backward and forward time, depending on t_to - t_from.
    """
    x = x_t.float()
    t = float(t_from)
    dt = (float(t_to) - float(t_from)) / float(n_steps)

    for _ in range(n_steps):
        k1 = pf_velocity(x, t, gmm, score_chunk=score_chunk)
        k2 = pf_velocity(x + 0.5 * dt * k1, t + 0.5 * dt, gmm, score_chunk=score_chunk)
        k3 = pf_velocity(x + 0.5 * dt * k2, t + 0.5 * dt, gmm, score_chunk=score_chunk)
        k4 = pf_velocity(x + dt * k3, t + dt, gmm, score_chunk=score_chunk)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
    return x


# -----------------------------
# Stability profile and grids
# -----------------------------

@torch.no_grad()
def diam0_of_mu(mu: torch.Tensor) -> float:
    D = torch.cdist(mu.float(), mu.float(), p=2.0)  # (K,K)
    return float(D.max().item())


def stability_profile_L(t: float, gmm: GMMOU, D0: float) -> float:
    alpha = math.exp(-gmm.beta * float(t))
    diam_t = alpha * float(D0)
    _, s2_t = gmm.gmm_ou_params(float(t))
    s2 = float(max(float(s2_t), 1e-12))
    term = abs(1.0 - 1.0 / s2) + (diam_t * diam_t) / (4.0 * s2 * s2)
    return float(gmm.beta) * term


def make_uniform_grid(T: float, n: int) -> List[float]:
    T = float(T)
    n = int(n)
    return [T * (1.0 - k / n) for k in range(0, n + 1)]


def make_stability_balanced_grid(
    T: float,
    n: int,
    gmm: GMMOU,
    D0: float,
    n_grid: int = 4001,
) -> List[float]:
    T = float(T)
    n = int(n)
    ts = np.linspace(0.0, T, int(n_grid), dtype=np.float64)
    Ls = np.array([stability_profile_L(float(t), gmm=gmm, D0=D0) for t in ts], dtype=np.float64)

    dts = ts[1:] - ts[:-1]
    A = np.zeros_like(ts)
    A[1:] = np.cumsum(0.5 * (Ls[1:] + Ls[:-1]) * dts)

    AT = float(A[-1])
    A_targets = np.array([(k / n) * AT for k in range(0, n + 1)], dtype=np.float64)

    t_inc = np.interp(A_targets, A, ts)
    t_dec = list(t_inc[::-1].tolist())
    t_dec[0] = T
    t_dec[-1] = 0.0
    return [float(x) for x in t_dec]


# -----------------------------
# Density for NLL at t=0
# -----------------------------

@torch.no_grad()
def log_prob_isotropic_gmm(x: torch.Tensor, mu: torch.Tensor, pi: torch.Tensor, s2: float) -> torch.Tensor:
    x = x.float()
    mu = mu.to(x.device).float()
    pi = pi.to(x.device).float().clamp_min(1e-30)

    x2 = (x * x).sum(dim=-1, keepdim=True)         # (N,1)
    m2 = (mu * mu).sum(dim=-1).view(1, -1)         # (1,K)
    xm = x @ mu.t()                                # (N,K)
    dist2 = (x2 + m2 - 2.0 * xm).clamp_min(0.0)    # (N,K)

    d = x.shape[-1]
    s2 = float(max(float(s2), 1e-12))
    log_norm = -0.5 * (d * math.log(2.0 * math.pi * s2))
    log_pi = torch.log(pi).view(1, -1)
    logits = log_pi + log_norm - 0.5 * dist2 / s2
    return torch.logsumexp(logits, dim=-1)


# -----------------------------
# Distillation, training, evaluation
# -----------------------------

@dataclass
class TrainCfg:
    device: str = "cuda"
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    depth: int = 4
    width: int = 512

    steps_total: int = 20000
    batch_size: int = 1024

    eval_n: int = 50000
    eval_batch: int = 1024

    teacher_steps_train_total: int = 200
    teacher_steps_eval_total: int = 400
    score_chunk: int = 0


def steps_for_segment(total_steps: int, n: int) -> int:
    return max(1, int(total_steps) // int(n))


def teacher_steps_for_segment(total_steps: int, T: float, t_from: float, t_to: float) -> int:
    dur = abs(float(t_to) - float(t_from))
    frac = dur / max(float(T), 1e-12)
    return max(5, int(round(total_steps * frac)))


@torch.no_grad()
def estimate_end_to_end_metrics(
    models: List[nn.Module],
    grid: List[float],
    gmm: GMMOU,
    T: float,
    cfg: TrainCfg,
    device: torch.device,
    eval_seed: int,
    deterministic: bool,
) -> Tuple[float, float]:
    for m in models:
        m.eval()

    set_seed(int(eval_seed), deterministic=deterministic)

    total_num = 0.0
    total_den = 0.0
    total_nll = 0.0
    seen = 0

    while seen < cfg.eval_n:
        n = min(cfg.eval_batch, cfg.eval_n - seen)
        xT = gmm.sample_xt(t=float(T), n=n, device=device)

        y = phi_t_to_s_rk4(
            xT,
            t_from=float(T),
            t_to=0.0,
            gmm=gmm,
            n_steps=int(cfg.teacher_steps_eval_total),
            score_chunk=int(cfg.score_chunk),
        )

        x = xT.float()
        for k in range(1, len(grid)):
            x = models[k - 1](x).float()
        yhat = x

        total_num += float((yhat - y).pow(2).sum(dim=-1).sum().detach().cpu())
        total_den += float(y.pow(2).sum(dim=-1).sum().detach().cpu())

        logp0 = log_prob_isotropic_gmm(yhat, mu=gmm.mu, pi=gmm.pi, s2=(gmm.sigma0 ** 2))
        total_nll += float((-logp0).sum().detach().cpu())

        seen += n

    relmse_map = total_num / max(total_den, 1e-12)
    nll0 = total_nll / max(cfg.eval_n, 1)
    return float(relmse_map), float(nll0)


def dump_grid(grid: List[float], name: str, sigma0: float, n: int) -> None:
    # grid is decreasing: [T, ..., 0]
    widths = [grid[k-1] - grid[k] for k in range(1, len(grid))]  # positive
    print(f"\n[{name}] sigma0={sigma0:g} n={n}")
    print("k    t_k (decreasing)        Δt_k")
    for k, t in enumerate(grid):
        if k == 0:
            dt = float("nan")
        else:
            dt = widths[k-1]
        print(f"{k:2d}   {t: .10e}    {dt: .10e}")


def train_one_run(
    gmm: GMMOU,
    T: float,
    grid: List[float],
    cfg: TrainCfg,
    trial_seed: int,
    eval_seed: int,
    deterministic: bool,
) -> Tuple[float, float, float, float]:
    """
    Trains one student per segment.

    Means mu and weights pi are fixed outside this function.

    trial_seed controls only training randomness
    - segment model initialization
    - minibatch sampling

    eval_seed controls only evaluation sampling, independent of trial_seed.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    nseg = len(grid) - 1

    models: List[nn.Module] = []
    worst_seg = 0.0

    t0 = time.time()

    for k in range(1, len(grid)):
        t_from = float(grid[k - 1])
        t_to = float(grid[k])

        init_seed_seg = seed_mix(int(trial_seed), 101, k)
        data_seed_seg = seed_mix(int(trial_seed), 202, k)

        set_seed(int(init_seed_seg), deterministic=deterministic)
        model = build_mlp(d=gmm.d, width=int(cfg.width), depth=int(cfg.depth)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        set_seed(int(data_seed_seg), deterministic=deterministic)

        seg_steps = steps_for_segment(cfg.steps_total, nseg)
        teacher_steps_seg = teacher_steps_for_segment(
            total_steps=int(cfg.teacher_steps_train_total),
            T=float(T),
            t_from=t_from,
            t_to=t_to,
        )

        model.train()
        for _ in range(seg_steps):
            x = gmm.sample_xt(t=t_from, n=int(cfg.batch_size), device=device)
            y = phi_t_to_s_rk4(
                x,
                t_from=t_from,
                t_to=t_to,
                gmm=gmm,
                n_steps=int(teacher_steps_seg),
                score_chunk=int(cfg.score_chunk),
            )
            yhat = model(x).float()
            loss = F.mse_loss(yhat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

        with torch.no_grad():
            xev = gmm.sample_xt(t=t_from, n=min(4096, int(cfg.eval_batch)), device=device)
            yev = phi_t_to_s_rk4(
                xev,
                t_from=t_from,
                t_to=t_to,
                gmm=gmm,
                n_steps=max(20, int(teacher_steps_seg)),
                score_chunk=int(cfg.score_chunk),
            )
            yhat_ev = model(xev).float()
            num = (yhat_ev - yev).pow(2).sum(dim=-1).mean()
            den = yev.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
            seg_rel = float((num / den).detach().cpu())
            worst_seg = max(worst_seg, seg_rel)

        models.append(model)

    train_sec = time.time() - t0

    relmse_map, nll0 = estimate_end_to_end_metrics(
        models=models,
        grid=grid,
        gmm=gmm,
        T=float(T),
        cfg=cfg,
        device=device,
        eval_seed=int(eval_seed),
        deterministic=deterministic,
    )

    return float(relmse_map), float(nll0), float(train_sec), float(worst_seg)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="out_exp4_stability_balanced_vs_uniform")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--deterministic", action="store_true")

    # fixed mixture means and weights
    parser.add_argument("--mu_seed", type=int, default=0)

    # evaluation sampling seed base, independent of training seeds
    parser.add_argument("--eval_seed_base", type=int, default=0)

    # GMM-OU
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--mu_box", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sigma_inf", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=1.0)

    # sweeps
    parser.add_argument("--sigmas", type=str, default="1")
    parser.add_argument("--ns", type=str, default="4,8,16")
    parser.add_argument("--grid_numerical_points", type=int, default=10001)

    # student
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)

    # training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps_total", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # teacher
    parser.add_argument("--teacher_steps_train_total", type=int, default=100)
    parser.add_argument("--teacher_steps_eval_total", type=int, default=100)
    parser.add_argument("--score_chunk", type=int, default=0)

    # eval
    parser.add_argument("--eval_n", type=int, default=50000)
    parser.add_argument("--eval_batch", type=int, default=1024)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    seeds = parse_csv_ints(args.seeds)
    sigmas = parse_csv_floats(args.sigmas)
    ns = parse_csv_ints(args.ns)

    cfg = TrainCfg(
        device=args.device,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        depth=int(args.depth),
        width=int(args.width),
        steps_total=int(args.steps_total),
        batch_size=int(args.batch_size),
        eval_n=int(args.eval_n),
        eval_batch=int(args.eval_batch),
        teacher_steps_train_total=int(args.teacher_steps_train_total),
        teacher_steps_eval_total=int(args.teacher_steps_eval_total),
        score_chunk=int(args.score_chunk),
    )

    # fixed mu and pi
    set_seed(int(args.mu_seed), deterministic=bool(args.deterministic))
    mu = (torch.rand(int(args.K), int(args.d)) * 2.0 - 1.0) * float(args.mu_box)
    pi = torch.ones(int(args.K)) / float(args.K)
    D0 = diam0_of_mu(mu)

    per_run_csv = os.path.join(args.out_dir, "per_run_results.csv")
    per_run_fields = [
        "exp", "grid", "n", "seed",
        "sigma0",
        "relmse_map", "nll0", "worst_seg_relmse",
        "train_sec",
        "beta", "sigma_inf", "mu_box", "d", "K", "T",
        "steps_total", "steps_per_seg", "batch_size", "lr",
        "teacher_steps_train_total", "teacher_steps_eval_total", "score_chunk",
        "grid_numerical_points",
        "D0",
        "mu_seed",
        "eval_seed",
    ]
    if not os.path.isfile(per_run_csv):
        write_csv(per_run_csv, [], per_run_fields)

    bucket_map: Dict[Tuple[str, float, int], List[float]] = defaultdict(list)
    bucket_nll: Dict[Tuple[str, float, int], List[float]] = defaultdict(list)

    plan: List[Tuple[float, int, int, str]] = []
    for sigma0 in sigmas:
        for n in ns:
            for seed in seeds:
                for grid_name in ["stability","uniform"]:
                    plan.append((float(sigma0), int(n), int(seed), grid_name))

    pbar = tqdm(plan, desc="exp_stability_grid_vs_uniform", dynamic_ncols=True)
    for sigma0, n, seed, grid_name in pbar:
        pbar.set_postfix({"sig": f"{sigma0:g}", "n": n, "grid": grid_name, "seed": seed})

        gmm = GMMOU(
            d=int(args.d),
            K=int(args.K),
            mu=mu,
            pi=pi,
            sigma0=float(sigma0),
            beta=float(args.beta),
            sigma_inf=float(args.sigma_inf),
        )

        if grid_name == "uniform":
            grid = make_uniform_grid(T=float(args.T), n=int(n))
        else:
            grid = make_stability_balanced_grid(
                T=float(args.T),
                n=int(n),
                gmm=gmm,
                D0=float(D0),
                n_grid=int(args.grid_numerical_points),
            )

        dump_grid(grid, grid_name, sigma0, n)

        eval_seed = seed_mix(int(args.eval_seed_base), int(n), int(round(sigma0 * 1e9)))

        relmse_map, nll0, train_sec, worst_seg = train_one_run(
            gmm=gmm,
            T=float(args.T),
            grid=grid,
            cfg=cfg,
            trial_seed=int(seed),
            eval_seed=int(eval_seed),
            deterministic=bool(args.deterministic),
        )

        row = {
            "exp": "exp_stability_grid_vs_uniform",
            "grid": grid_name,
            "n": int(n),
            "seed": int(seed),
            "sigma0": float(sigma0),
            "relmse_map": float(relmse_map),
            "nll0": float(nll0),
            "worst_seg_relmse": float(worst_seg),
            "train_sec": float(train_sec),
            "beta": float(args.beta),
            "sigma_inf": float(args.sigma_inf),
            "mu_box": float(args.mu_box),
            "d": int(args.d),
            "K": int(args.K),
            "T": float(args.T),
            "steps_total": int(cfg.steps_total),
            "steps_per_seg": int(steps_for_segment(cfg.steps_total, int(n))),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "teacher_steps_train_total": int(cfg.teacher_steps_train_total),
            "teacher_steps_eval_total": int(cfg.teacher_steps_eval_total),
            "score_chunk": int(cfg.score_chunk),
            "grid_numerical_points": int(args.grid_numerical_points),
            "D0": float(D0),
            "mu_seed": int(args.mu_seed),
            "eval_seed": int(eval_seed),
        }

        with open(per_run_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_run_fields)
            w.writerow(row)

        bucket_map[(grid_name, float(sigma0), int(n))].append(float(relmse_map))
        bucket_nll[(grid_name, float(sigma0), int(n))].append(float(nll0))

    long_rows: List[Dict[str, Any]] = []
    for grid_name in ["uniform", "stability"]:
        for sigma0 in sigmas:
            for n in ns:
                xs = bucket_map[(grid_name, float(sigma0), int(n))]
                ys = bucket_nll[(grid_name, float(sigma0), int(n))]
                m, sd = mean_std(xs)
                mn, sdn = mean_std(ys)
                long_rows.append(
                    {
                        "grid": grid_name,
                        "sigma0": f"{sigma0:g}",
                        "n": str(int(n)),
                        "relmse_mean±std": fmt_mean_std_sci2(m, sd),
                        "relmse_mean": f"{m:.8e}",
                        "relmse_std": f"{sd:.8e}",
                        "nll0_mean±std": fmt_mean_std_sci2(mn, sdn),
                        "nll0_mean": f"{mn:.8e}",
                        "nll0_std": f"{sdn:.8e}",
                        "n_trials": str(len(xs)),
                    }
                )

    long_rows = sorted(long_rows, key=lambda r: (r["grid"], float(r["sigma0"]), int(r["n"])))

    write_csv(
        os.path.join(args.out_dir, "agg_long.csv"),
        long_rows,
        fieldnames=[
            "grid", "sigma0", "n",
            "relmse_mean±std", "relmse_mean", "relmse_std",
            "nll0_mean±std", "nll0_mean", "nll0_std",
            "n_trials",
        ],
    )

    for grid_name in ["uniform", "stability"]:
        src = [
            {"sigma0": r["sigma0"], "n": r["n"], "cell": r["relmse_mean±std"]}
            for r in long_rows
            if r["grid"] == grid_name
        ]
        piv = pivot_table(
            src,
            row_key="sigma0",
            col_key="n",
            val_key="cell",
            row_order=[f"{s:g}" for s in sigmas],
            col_order=[str(int(n)) for n in ns],
        )
        write_csv(
            os.path.join(args.out_dir, f"agg_pivot_{grid_name}.csv"),
            piv,
            fieldnames=["sigma0"] + [str(int(n)) for n in ns],
        )

    print("\nSaved outputs")
    print(f"  {per_run_csv}")
    print(f"  {os.path.join(args.out_dir, 'agg_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'agg_pivot_uniform.csv')}")
    print(f"  {os.path.join(args.out_dir, 'agg_pivot_stability.csv')}")
    print("\nDone")


if __name__ == "__main__":
    main()
