# A Quantitative Approximation Framework for Flow Distillation in Diffusion Models

PyTorch experiments for score approximation and probability-flow map distillation on isotropic Gaussian mixture models under an Ornstein-Uhlenbeck flow.

## Files

- `exp1_static_approximation.py`: static score approximation; sweeps `sigma0` and width.
- `exp2_flow_map_bound_sweep_aligned.py`: one-step flow-map distillation; sweeps `sigma0` and bound `B`.
- `exp3_flow_map_blocks_sweep_T1_sigma1e4.py`: full `T=1` flow-map distillation; sweeps block count.
- `exp4_stability_balanced_vs_uniform.py`: compares uniform and stability-balanced grids.

## Requirements

```bash
pip install numpy tqdm matplotlib torch
```

CUDA is recommended for the default runs.

## Run

```bash
python exp1_static_approximation.py --resume
python exp2_flow_map_bound_sweep_aligned.py --resume
python exp3_flow_map_blocks_sweep_T1_sigma1e4.py --resume
python exp4_stability_balanced_vs_uniform.py
```

## Outputs

Default output directories:

```text
out_exp1_static_approximation/
out_exp2_flow_map_bound/
out_exp3_flow_map_blocks/
out_exp4_stability_balanced_vs_uniform/
```

Each output directory contains `per_run_results.csv` and aggregate CSV files. Experiments 1-3 also use `cache_runs/` for `--resume`.

To avoid overwriting or appending to included results, pass a different `--out_dir`.
