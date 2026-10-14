# Figure 8 assumption ablations

`config.json` defines 19 conditions and ten paired seeds (190 completed runs).
The six families test class-dependent entropy, conditional dependence,
non-exponential latents, signed latents, output activation, and weight decay.

Aggregate generated run records:

```bash
python Codes/fig8-ablation_test/fig8.py aggregate \
  --plan Codes/fig8-ablation_test/config.json \
  --results-root Data/fig8/runs \
  --output-dir Data/fig8/report
```

The report contains `runs.csv`, `conditions.csv`, `paired.csv`,
`paired-summary.csv`, `fig8.pdf/png`, `caption.txt`, `README.md`, and
`manifest.json`.

To rerun the simulations, first create the shared data-generating process,
then run the full grid (or pass `--conditions` and `--seeds` for a subset):

```bash
python Codes/fig8-ablation_test/fig8.py prepare-dgp \
  --plan Codes/fig8-ablation_test/config.json \
  --artifact-dir Data/fig8/dgp --device cpu
python Codes/fig8-ablation_test/fig8.py simulate \
  --plan Codes/fig8-ablation_test/config.json \
  --dgp-path Data/fig8/dgp \
  --output-root Data/fig8/runs --device cpu --resume
```


