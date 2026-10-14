# Figure 7 reliability analysis

`fig7.py` is the self-contained entry point. It imports the shared numerical
helpers in `Codes/utils.py`, writes scientific results to `Data/fig7`, and
does not require an experiment-tracking service.
The saved design contains 30 baseline runs and 40 paired reliability runs
(10 initialization seeds by 4 weight-decay values).

Recreate the report from saved runs:

```bash
python Codes/fig7-weight_decay_reliability/fig7.py plot
```

For new training, run these commands from the repository root:

```bash
python Codes/fig7-weight_decay_reliability/fig7.py baseline-sweep \
  --init-seeds 0:10
python Codes/fig7-weight_decay_reliability/fig7.py prepare-dgp \
  --artifact-dir Data/fig7/dgp/sparse --device cpu
python Codes/fig7-weight_decay_reliability/fig7.py sweep \
  --output-root Data/fig7/reliability \
  --dgp-path Data/fig7/dgp/sparse \
  --init-seeds 0:10 --weight-decays 0,0.003,0.01,0.03 --resume
```

The sweep is sequential and resumable; select a smaller seed or weight-decay
set for a quick check. Generated checkpoints are ignored by Git.

