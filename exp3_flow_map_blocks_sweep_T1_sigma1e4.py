#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flow_map_blocks_sweep_T1_sigma1e4.py

Compares residual and transformer student maps for learning the full backward
probability-flow transport Phi_{0<-T} on a fixed isotropic GMM-OU problem.

Sweep
- number of student blocks
- architecture in {resnet, transformer}
- averaged over training seeds

Outputs
- per_run_results.csv
- agg_long.csv
- agg_pivot.csv
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
# Model: ReLU MLP and flow-map students
# -----------------------------
def build_mlp(d: int, width: int, depth: int, act0: str = "relu") -> nn.Module:
    """
    Build the inner student MLP used by the residual map.
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


# Backward-compatible name for the residual map helper.
build_alt_mlp = build_mlp


# -----------------------------
# Student A: Definition-10-style residual composition
# -----------------------------
class ResNetFlowMapDef10(nn.Module):
    """
    x <- x + alpha * v_k(x)
    v_k(x) = -beta * (x + s_k(x))
    alpha = -T / n_blocks
    Each block has its own s_k, no time conditioning
    """
    def __init__(
        self,
        d: int,
        n_blocks: int,
        beta: float,
        T: float,
        inner_width: int,
        inner_depth: int,
        act0: str = "relu",
    ) -> None:
        super().__init__()
        self.d = d
        self.n_blocks = int(n_blocks)
        self.beta = float(beta)
        self.T = float(T)
        self.alpha = -self.T / float(self.n_blocks)

        self.s_nets = nn.ModuleList(
            [build_mlp(d=d, width=inner_width, depth=inner_depth, act0=act0)
             for _ in range(self.n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for s_net in self.s_nets:
            s = s_net(x).float()
            v = -self.beta * (x + s)
            x = x + self.alpha * v
        return x


# -----------------------------
# Student B: Transformer map with L attention blocks
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TransformerFlowMap(nn.Module):
    """
    Treat x in R^d as a length-d token sequence with scalar tokens.
    Embed scalar -> d_model, add learned pos emb, run L blocks, project back to scalar.
    """
    def __init__(
        self,
        d: int,
        n_blocks: int,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.n_blocks = int(n_blocks)
        self.d_model = int(d_model)

        self.in_proj = nn.Linear(1, self.d_model)
        self.pos = nn.Parameter(torch.zeros(self.d, self.d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.d_model, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout)
             for _ in range(self.n_blocks)]
        )

        self.out_proj = nn.Linear(self.d_model, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        xtok = x.unsqueeze(-1)                      # (B, d, 1)
        h = self.in_proj(xtok)                      # (B, d, d_model)
        h = h + self.pos.unsqueeze(0)               # (B, d, d_model)
        for blk in self.blocks:
            h = blk(h)
        y = self.out_proj(h).squeeze(-1)            # (B, d)
        return y


# -----------------------------
# GMM-OU, identical to your suite 2
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
        s2_t = (alpha * alpha) * (self.sigma0**2) + (1.0 - alpha * alpha) * (self.sigma_inf**2)
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
            xb = x[i : i + chunk]
            gamma = self.responsibilities(xb, t)
            bar_m = gamma @ m_t
            outs.append(-(xb.float() - bar_m) / float(s2_t))
        return torch.cat(outs, dim=0)


# -----------------------------
# Teacher map Phi_{0<-T}
# -----------------------------
def pf_velocity(x: torch.Tensor, t: float, gmm: GMMOU, score_chunk: int = 0) -> torch.Tensor:
    score = gmm.score_closed_form(x, t=t, chunk=score_chunk)
    return -gmm.beta * (x.float() + score)


@torch.no_grad()
def phi_0_from_T_rk4(x_T: torch.Tensor, T: float, gmm: GMMOU, n_steps: int, score_chunk: int = 0) -> torch.Tensor:
    x = x_T.float()
    dt = -T / float(n_steps)
    t = float(T)
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
    eval_n: int = 50000
    eval_batch: int = 1024
    score_chunk: int = 0
    teacher_steps: int = 200


@torch.no_grad()
def estimate_relmse_map(model: nn.Module, gmm: GMMOU, T: float, cfg: TrainCfg, device: torch.device) -> float:
    model.eval()
    total_num = 0.0
    total_den = 0.0
    seen = 0
    while seen < cfg.eval_n:
        n = min(cfg.eval_batch, cfg.eval_n - seen)
        x = gmm.sample_xt(t=T, n=n, device=device)
        y = phi_0_from_T_rk4(x, T=T, gmm=gmm, n_steps=cfg.teacher_steps, score_chunk=cfg.score_chunk)
        yhat = model(x).float()
        total_num += float((yhat - y).pow(2).sum(dim=-1).sum().detach().cpu())
        total_den += float((y).pow(2).sum(dim=-1).sum().detach().cpu())
        seen += n
    return total_num / max(total_den, 1e-12)


def build_student(args: argparse.Namespace, d: int, n_blocks: int) -> nn.Module:
    if args.arch == "resnet":
        return ResNetFlowMapDef10(
            d=d,
            n_blocks=n_blocks,
            beta=float(args.beta),
            T=float(args.T),
            inner_width=int(args.inner_width),
            inner_depth=int(args.inner_depth),
        )
    if args.arch == "transformer":
        return TransformerFlowMap(
            d=d,
            n_blocks=n_blocks,
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            ff_mult=int(args.ff_mult),
            dropout=float(args.dropout),
        )
    raise ValueError(f"Unknown arch {args.arch}")


def train_one(gmm: GMMOU, T: float, n_blocks: int, cfg: TrainCfg, args: argparse.Namespace, verbose: bool = False) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_student(args, d=gmm.d, n_blocks=n_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        x = gmm.sample_xt(t=T, n=cfg.batch_size, device=device)
        y = phi_0_from_T_rk4(x, T=T, gmm=gmm, n_steps=cfg.teacher_steps, score_chunk=cfg.score_chunk)
        yhat = model(x).float()
        loss = F.mse_loss(yhat, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if verbose and (step == 1 or step % 500 == 0):
            with torch.no_grad():
                num = (yhat - y).pow(2).sum(dim=-1).mean()
                den = y.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
                rel = float((num / den).detach().cpu())
            print(f"[{args.arch} L={n_blocks}] step {step:6d}/{cfg.steps}  loss={loss.item():.4e}  relmse(batch)={rel:.4e}")

    train_time = time.time() - t0
    eval_rel = estimate_relmse_map(model, gmm, T=T, cfg=cfg, device=device)
    return {"relmse": float(eval_rel), "train_sec": float(train_time), "device_is_cuda": 1.0 if device.type == "cuda" else 0.0}


def run_id(args: argparse.Namespace, seed: int, n_blocks: int) -> str:
    return (
        f"exp_flow_map_{args.arch}"
        f"_seed{seed}"
        f"_museed{int(args.mu_seed)}"
        f"_sig{float(args.sigma0):g}"
        f"_T{float(args.T):g}"
        f"_beta{float(args.beta):g}"
        f"_L{int(n_blocks)}"
        f"_steps{int(args.steps)}"
        f"_bs{int(args.batch_size)}"
        f"_lr{float(args.lr):g}"
        f"_d{int(args.d)}_K{int(args.K)}"
        f"_tsteps{int(args.teacher_steps)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", type=str, default="out_exp3_flow_map_blocks")
    p.add_argument("--arch", type=str, default="transformer", choices=["resnet", "transformer"])
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--blocks", type=str, default="1,2,4,8,16")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--d", type=int, default=64)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--mu_box", type=float, default=5.0)
    p.add_argument("--mu_seed", type=int, default=0)

    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--sigma_inf", type=float, default=1.0)

    p.add_argument("--sigma0", type=float, default=1e-3)
    p.add_argument("--T", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_n", type=int, default=50000)
    p.add_argument("--eval_batch", type=int, default=1024)
    p.add_argument("--teacher_steps", type=int, default=100)
    p.add_argument("--score_chunk", type=int, default=0)

    # ResNet inner nets
    p.add_argument("--inner_depth", type=int, default=2)
    p.add_argument("--inner_width", type=int, default=128)

    # Transformer hyperparameters
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    args = p.parse_args()

    ensure_dir(args.out_dir)
    cache_dir = os.path.join(args.out_dir, "cache_runs")
    ensure_dir(cache_dir)

    seeds = parse_csv_ints(args.seeds)
    blocks_list = parse_csv_ints(args.blocks)

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

    # Fixed mixture means
    g_mu = torch.Generator()
    g_mu.manual_seed(int(args.mu_seed))
    mu_fixed = (torch.rand(int(args.K), int(args.d), generator=g_mu) * 2.0 - 1.0) * float(args.mu_box)
    pi_fixed = torch.ones(int(args.K)) / float(args.K)

    per_run_csv = os.path.join(args.out_dir, "per_run_results.csv")
    per_run_fields = [
        "exp", "arch", "seed", "mu_seed",
        "sigma0", "T", "beta", "sigma_inf",
        "n_blocks",
        "inner_depth", "inner_width",
        "d_model", "n_heads", "ff_mult", "dropout",
        "relmse", "train_sec",
        "mu_box", "d", "K",
        "steps", "batch_size", "lr",
        "eval_n", "eval_batch",
        "teacher_steps", "score_chunk",
    ]
    if not os.path.isfile(per_run_csv):
        write_csv(per_run_csv, [], per_run_fields)

    bucket: Dict[int, List[float]] = defaultdict(list)

    plan: List[Tuple[int, int]] = [(seed, L) for seed in seeds for L in blocks_list]
    pbar = tqdm(plan, desc=f"exp_flow_map_{args.arch}", dynamic_ncols=True)

    for seed, L in pbar:
        set_seed(int(seed), deterministic=bool(args.deterministic))

        gmm = GMMOU(
            d=int(args.d),
            K=int(args.K),
            mu=mu_fixed,
            pi=pi_fixed,
            sigma0=float(args.sigma0),
            beta=float(args.beta),
            sigma_inf=float(args.sigma_inf),
        )

        rid = run_id(args, seed=int(seed), n_blocks=int(L))
        cache_path = os.path.join(cache_dir, f"{rid}.csv")

        metric = maybe_load_metric(cache_path) if bool(args.resume) else None
        if metric is None:
            pbar.set_postfix({"seed": seed, "L": L})
            met = train_one(gmm=gmm, T=float(args.T), n_blocks=int(L), cfg=cfg, args=args, verbose=False)

            metric = {
                "seed": float(seed),
                "mu_seed": float(args.mu_seed),
                "sigma0": float(args.sigma0),
                "T": float(args.T),
                "beta": float(args.beta),
                "sigma_inf": float(args.sigma_inf),
                "n_blocks": float(L),
                "inner_depth": float(args.inner_depth),
                "inner_width": float(args.inner_width),
                "d_model": float(args.d_model),
                "n_heads": float(args.n_heads),
                "ff_mult": float(args.ff_mult),
                "dropout": float(args.dropout),
                "relmse": float(met["relmse"]),
                "train_sec": float(met["train_sec"]),
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
            row["exp"] = f"exp_flow_map_{args.arch}"
            row["arch"] = args.arch
            with open(per_run_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=per_run_fields)
                w.writerow(row)

        bucket[int(L)].append(float(metric["relmse"]))

    long_rows: List[Dict[str, Any]] = []
    for L in blocks_list:
        xs = bucket[int(L)]
        m, sd = mean_std(xs)
        long_rows.append(
            {
                "arch": args.arch,
                "n_blocks": str(L),
                "mean±std": fmt_mean_std_sci2(m, sd),
                "mean": f"{m:.8e}",
                "std": f"{sd:.8e}",
                "n_trials": str(len(xs)),
            }
        )
    long_rows = sorted(long_rows, key=lambda r: int(r["n_blocks"]))

    pivot_src = [{"n_blocks": r["n_blocks"], "col": "relmse", "cell": r["mean±std"]} for r in long_rows]
    pivot_rows = pivot_table(
        pivot_src,
        row_key="n_blocks",
        col_key="col",
        val_key="cell",
        row_order=[str(L) for L in blocks_list],
        col_order=["relmse"],
    )

    write_csv(os.path.join(args.out_dir, "agg_long.csv"), long_rows,
              fieldnames=["arch", "n_blocks", "mean±std", "mean", "std", "n_trials"])
    write_csv(os.path.join(args.out_dir, "agg_pivot.csv"), pivot_rows,
              fieldnames=["n_blocks", "relmse"])

    print("\nSaved outputs")
    print(os.path.join(args.out_dir, "agg_long.csv"))
    print(os.path.join(args.out_dir, "agg_pivot.csv"))
    print(os.path.join(args.out_dir, "per_run_results.csv"))
    print("\nDone")


if __name__ == "__main__":
    main()
