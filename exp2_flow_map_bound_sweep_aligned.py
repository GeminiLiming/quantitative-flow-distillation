#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flow_map_bound_sweep_aligned.py

Aligns architecture and data generation with static_approximation_two_exps.py

Task
- Learn a student map Psi(x_Delta) ≈ Phi_{0<-Delta}(x_Delta)
- Delta is the computable OU half-life time Delta = log(2)/(2*beta)
- Sweep parameter bound B by hard clipping all Linear weights to [-B, B]

Change requested
- The trial seed affects only stochasticity in sampling and training (model init, minibatches, etc.)
- Mixture means are fixed across all trials and configurations

Outputs
- per_run_results.csv
- agg_long.csv
- agg_pivot.csv

Notes
- Student architecture uses build_mlp below
- Data generation uses GMMOU.sample_xt below
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


# -----------------------------
# Model: alternating ReLU / ReQU
# -----------------------------


class ReQU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)


def build_mlp(d: int, width: int, depth: int, act0: str = "relu") -> nn.Module:
    """
    depth = number of hidden layers
    activation alternates ReLU, ReQU, ReLU, ReQU, ...
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


@torch.no_grad()
def clip_linear_params_(model: nn.Module, B: float) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.clamp_(-B, B)


# -----------------------------
# GMM-OU: sampling + closed-form score
# -----------------------------


@dataclass
class GMMOU:
    d: int
    K: int
    mu: torch.Tensor
    pi: torch.Tensor
    sigma0: float
    beta: float
    sigma_inf: float = 1.0

    def gmm_ou_params(self, t: float) -> Tuple[torch.Tensor, float]:
        alpha = math.exp(-self.beta * t)
        m_t = self.mu * alpha
        s2_t = (alpha * alpha) * (self.sigma0**2) + (1.0 - alpha * alpha) * (
            self.sigma_inf**2
        )
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
        x2 = (x * x).sum(dim=-1, keepdim=True)
        m2 = (m_t * m_t).sum(dim=-1).view(1, -1)
        xm = x @ m_t.t()
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
    def score_closed_form(
        self, x: torch.Tensor, t: float, chunk: int = 0
    ) -> torch.Tensor:
        m_t, s2_t = self.gmm_ou_params(t)
        m_t = m_t.to(x.device).float()

        if chunk is None or chunk <= 0:
            gamma = self.responsibilities(x, t)
            bar_m = gamma @ m_t
            return -(x.float() - bar_m) / float(s2_t)

        outs = []
        N = x.shape[0]
        for i in range(0, N, chunk):
            xb = x[i : i + chunk]
            gamma = self.responsibilities(xb, t)
            bar_m = gamma @ m_t
            outs.append(-(xb.float() - bar_m) / float(s2_t))
        return torch.cat(outs, dim=0)


# -----------------------------
# Teacher map Phi_{0<-Delta}
# -----------------------------


def pf_velocity(
    x: torch.Tensor, t: float, gmm: GMMOU, score_chunk: int = 0
) -> torch.Tensor:
    score = gmm.score_closed_form(x, t=t, chunk=score_chunk)
    return -gmm.beta * (x.float() + score)


@torch.no_grad()
def phi_0_from_delta_rk4(
    x_delta: torch.Tensor,
    delta: float,
    gmm: GMMOU,
    n_steps: int,
    score_chunk: int = 0,
) -> torch.Tensor:
    x = x_delta.float()
    dt = -delta / float(n_steps)

    t = float(delta)
    for _ in range(n_steps):
        k1 = pf_velocity(x, t, gmm, score_chunk=score_chunk)
        k2 = pf_velocity(x + 0.5 * dt * k1, t + 0.5 * dt, gmm, score_chunk=score_chunk)
        k3 = pf_velocity(x + 0.5 * dt * k2, t + 0.5 * dt, gmm, score_chunk=score_chunk)
        k4 = pf_velocity(x + dt * k3, t + dt, gmm, score_chunk=score_chunk)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
    return x


# -----------------------------
# Training and eval
# -----------------------------


@dataclass
class TrainCfg:
    device: str = "cuda"
    lr: float = 2e-4
    weight_decay: float = 0.0
    steps: int = 20000
    batch_size: int = 1024
    grad_clip: float = 1.0
    eval_n: int = 20000
    eval_batch: int = 1024
    score_chunk: int = 0
    teacher_steps: int = 200


@torch.no_grad()
def estimate_relmse_map(
    model: nn.Module,
    gmm: GMMOU,
    delta: float,
    cfg: TrainCfg,
    device: torch.device,
) -> float:
    model.eval()
    total_num = 0.0
    total_den = 0.0
    seen = 0
    while seen < cfg.eval_n:
        n = min(cfg.eval_batch, cfg.eval_n - seen)
        x = gmm.sample_xt(t=delta, n=n, device=device)
        y = phi_0_from_delta_rk4(
            x,
            delta=delta,
            gmm=gmm,
            n_steps=cfg.teacher_steps,
            score_chunk=cfg.score_chunk,
        )
        yhat = model(x).float()

        total_num += float((yhat - y).pow(2).sum(dim=-1).sum().detach().cpu())
        total_den += float((y).pow(2).sum(dim=-1).sum().detach().cpu())
        seen += n
    return total_num / max(total_den, 1e-12)


def train_for_bound(
    gmm: GMMOU,
    delta: float,
    B: float,
    cfg: TrainCfg,
    width: int,
    depth: int,
    verbose: bool = False,
) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_mlp(d=gmm.d, width=width, depth=depth).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        x = gmm.sample_xt(t=delta, n=cfg.batch_size, device=device)
        y = phi_0_from_delta_rk4(
            x,
            delta=delta,
            gmm=gmm,
            n_steps=cfg.teacher_steps,
            score_chunk=cfg.score_chunk,
        )
        yhat = model(x).float()
        loss = F.mse_loss(yhat, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        clip_linear_params_(model, B)

        if verbose and (step == 1 or step % 500 == 0):
            with torch.no_grad():
                num = (yhat - y).pow(2).sum(dim=-1).mean()
                den = y.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
                rel = float((num / den).detach().cpu())
            print(
                f"[B={B:g}] step {step:6d}/{cfg.steps}  loss={loss.item():.4e}  relmse(batch)={rel:.4e}"
            )

    train_time = time.time() - t0
    device = next(model.parameters()).device
    eval_rel = estimate_relmse_map(model, gmm, delta=delta, cfg=cfg, device=device)

    return {
        "relmse": float(eval_rel),
        "train_sec": float(train_time),
        "device_is_cuda": 1.0 if device.type == "cuda" else 0.0,
    }


# -----------------------------
# Caching
# -----------------------------


def run_id(
    exp_name: str,
    seed: int,
    mu_seed: int,
    sigma0: float,
    depth: int,
    width: int,
    B: float,
    delta: float,
    steps: int,
    batch: int,
    lr: float,
    beta: float,
    mu_box: float,
    d: int,
    K: int,
    teacher_steps: int,
) -> str:
    return (
        f"{exp_name}"
        f"_seed{seed}"
        f"_museed{mu_seed}"
        f"_sig{sigma0:g}"
        f"_dep{depth}"
        f"_wid{width}"
        f"_B{B:g}"
        f"_del{delta:.6f}"
        f"_steps{steps}"
        f"_bs{batch}"
        f"_lr{lr:g}"
        f"_beta{beta:g}"
        f"_box{mu_box:g}"
        f"_d{d}_K{K}"
        f"_tsteps{teacher_steps}"
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
# Driver
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="out_exp2_flow_map_bound")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--mu_box", type=float, default=5.0)
    parser.add_argument(
        "--mu_seed",
        type=int,
        default=0,
        help="seed used only to generate fixed mixture means",
    )
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sigma_inf", type=float, default=1.0)

    parser.add_argument("--sigmas", type=str, default="0.1, 0.01, 0.001, 0.0001")
    parser.add_argument("--bounds_B", type=str, default="0.001, 0.01, 0.1, 0.25")

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_n", type=int, default=50000)
    parser.add_argument("--eval_batch", type=int, default=1024)

    parser.add_argument("--teacher_steps", type=int, default=10)
    parser.add_argument("--score_chunk", type=int, default=0)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cache_dir = os.path.join(args.out_dir, "cache_runs")
    ensure_dir(cache_dir)

    seeds = parse_csv_ints(args.seeds)
    sigmas = parse_csv_floats(args.sigmas)
    bounds_B = parse_csv_floats(args.bounds_B)

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
        teacher_steps=int(args.teacher_steps),
    )

    delta = math.log(2.0) / (2.0 * float(args.beta))

    # -------------------------
    # Fixed mixture means, independent of trial seeds
    # -------------------------
    g_mu = torch.Generator()
    g_mu.manual_seed(int(args.mu_seed))
    mu_fixed = (
        torch.rand(int(args.K), int(args.d), generator=g_mu) * 2.0 - 1.0
    ) * float(args.mu_box)
    pi_fixed = torch.ones(int(args.K)) / float(args.K)

    per_run_csv = os.path.join(args.out_dir, "per_run_results.csv")
    per_run_fields = [
        "exp",
        "seed",
        "mu_seed",
        "sigma0",
        "depth",
        "width",
        "B",
        "delta",
        "relmse",
        "train_sec",
        "beta",
        "sigma_inf",
        "mu_box",
        "d",
        "K",
        "steps",
        "batch_size",
        "lr",
        "eval_n",
        "eval_batch",
        "teacher_steps",
        "score_chunk",
    ]
    if not os.path.isfile(per_run_csv):
        write_csv(per_run_csv, [], per_run_fields)

    bucket: Dict[Tuple[float, float], List[float]] = defaultdict(list)

    plan: List[Tuple[int, float, float]] = []
    for seed in seeds:
        for sigma0 in sigmas:
            for B in bounds_B:
                plan.append((int(seed), float(sigma0), float(B)))

    pbar = tqdm(plan, desc="exp_flow_map_bound", dynamic_ncols=True)
    for seed, sigma0, B in pbar:
        # Seed only affects stochastic parts (model init, minibatches, sampling),
        # while mu_fixed is already locked by mu_seed.
        set_seed(int(seed), deterministic=bool(args.deterministic))

        gmm = GMMOU(
            d=int(args.d),
            K=int(args.K),
            mu=mu_fixed,
            pi=pi_fixed,
            sigma0=float(sigma0),
            beta=float(args.beta),
            sigma_inf=float(args.sigma_inf),
        )

        rid = run_id(
            exp_name="exp_flow_map_bound",
            seed=int(seed),
            mu_seed=int(args.mu_seed),
            sigma0=float(sigma0),
            depth=int(args.depth),
            width=int(args.width),
            B=float(B),
            delta=float(delta),
            steps=int(cfg.steps),
            batch=int(cfg.batch_size),
            lr=float(cfg.lr),
            beta=float(args.beta),
            mu_box=float(args.mu_box),
            d=int(args.d),
            K=int(args.K),
            teacher_steps=int(cfg.teacher_steps),
        )
        cache_path = os.path.join(cache_dir, f"{rid}.csv")

        metric = maybe_load_metric(cache_path) if bool(args.resume) else None
        if metric is None:
            pbar.set_postfix({"seed": seed, "sig": f"{sigma0:g}", "B": f"{B:g}"})
            met = train_for_bound(
                gmm=gmm,
                delta=float(delta),
                B=float(B),
                cfg=cfg,
                width=int(args.width),
                depth=int(args.depth),
                verbose=False,
            )
            metric = {
                "seed": float(seed),
                "mu_seed": float(args.mu_seed),
                "sigma0": float(sigma0),
                "depth": float(args.depth),
                "width": float(args.width),
                "B": float(B),
                "delta": float(delta),
                "relmse": float(met["relmse"]),
                "train_sec": float(met["train_sec"]),
                "beta": float(args.beta),
                "sigma_inf": float(args.sigma_inf),
                "mu_box": float(args.mu_box),
                "d": float(args.d),
                "K": float(args.K),
                "steps": float(cfg.steps),
                "batch_size": float(cfg.batch_size),
                "lr": float(cfg.lr),
                "eval_n": float(cfg.eval_n),
                "eval_batch": float(cfg.eval_batch),
                "teacher_steps": float(cfg.teacher_steps),
                "score_chunk": float(cfg.score_chunk),
            }
            save_metric(cache_path, metric)

            row = dict(metric)
            row["exp"] = "exp_flow_map_bound"
            with open(per_run_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=per_run_fields)
                w.writerow(row)

        bucket[(float(sigma0), float(B))].append(float(metric["relmse"]))

    long_rows: List[Dict[str, Any]] = []
    for sigma0 in sigmas:
        for B in bounds_B:
            xs = bucket[(float(sigma0), float(B))]
            m, sd = mean_std(xs)
            long_rows.append(
                {
                    "sigma0": f"{sigma0:g}",
                    "B": f"{B:g}",
                    "mean": f"{m:.8e}",
                    "std": f"{sd:.8e}",
                    "mean±std": fmt_mean_std_sci2(m, sd),
                    "n_trials": str(len(xs)),
                    "depth": str(args.depth),
                    "width": str(args.width),
                    "delta": f"{delta:.8e}",
                    "teacher_steps": str(cfg.teacher_steps),
                    "mu_seed": str(args.mu_seed),
                }
            )
    long_rows = sorted(long_rows, key=lambda r: (float(r["sigma0"]), float(r["B"])))

    pivot_src = [
        {"sigma0": r["sigma0"], "B": r["B"], "cell": r["mean±std"]} for r in long_rows
    ]
    pivot_rows = pivot_table(
        pivot_src,
        row_key="sigma0",
        col_key="B",
        val_key="cell",
        row_order=[f"{s:g}" for s in sigmas],
        col_order=[f"{b:g}" for b in bounds_B],
    )

    write_csv(
        os.path.join(args.out_dir, "agg_long.csv"),
        long_rows,
        fieldnames=[
            "sigma0",
            "B",
            "mean±std",
            "mean",
            "std",
            "n_trials",
            "depth",
            "width",
            "delta",
            "teacher_steps",
            "mu_seed",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "agg_pivot.csv"),
        pivot_rows,
        fieldnames=["sigma0"] + [f"{b:g}" for b in bounds_B],
    )

    print("\nSaved outputs")
    print(f"  {os.path.join(args.out_dir, 'agg_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'agg_pivot.csv')}")
    print(f"  {os.path.join(args.out_dir, 'per_run_results.csv')}")
    print("\nDone")


if __name__ == "__main__":
    main()
