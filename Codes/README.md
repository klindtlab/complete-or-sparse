# Code entry points

The root `README.md` contains the complete figure/data table and exact
commands. The main entry points are:

- `generate-traces.py` and `plot-traces.py`: Figures 1, 5, and 6;
- `fig2&9-capacity_phase_transition/`: Figures 2 and 9;
- `fig3-compressed_sensing_recovery/`: Figure 3;
- `fig4-cross_system_comparison/`: external-data preparation and Figure 4;
- `fig4_mis.py`: minimal MIS functions used by the Figure 4 vision notebook;
- `fig7-weight_decay_reliability/fig7.py`: Figure 7;
- `fig8-ablation_test/fig8.py`: Figure 8;
- `utils.py`: shared numerical functions only.

All default scientific inputs and outputs live under `Data/`. No SLURM files,
cluster launchers, experiment-tracking clients, or required precomputed traces
are part of the reproduction path.
