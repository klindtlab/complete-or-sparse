# Figure 1 simulation

`train.py` contains one simulation run and uses the shared networks, sphere
sampler, and recovery metrics in `Codes/utils.py`. The repository-level
`generate-traces.py` executes the complete 40-run paper grid; `plot-traces.py`
renders the result.

Run the full grid from the repository root with:

```bash
python Codes/generate-traces.py fig1
python Codes/plot-traces.py fig1
```

For a single small interface check:

```bash
python 'Codes/fig1&5&6-axis_aligned_neurons/train.py' \
  --seed 0 --n-epochs 10 --n-log-steps 10 \
  --nr-indices 1000 --batch-size 100 \
  --expon 1 --relu_enc 1
```

The paper runs use much larger step budgets. Outputs go to
`Data/fig1/runs/<configuration>` by default. Model checkpoints are written
only when `--save-checkpoints` is requested.
