# Complete or Sparse
### A Tale of Two Identifiabilities

<!-- TODO: author list with links -->
Junze Zhou, David Klindt

**Abstract:** Understanding when neural networks learn interpretable representations is central to mechanistic interpretability. We study this through identifiability: when can supervised learning recover the true latent variables underlying the data? We introduce two *data generating processes* distinguished by their *causal direction* where either labels cause latents (*generative*) or latents cause labels (*discriminative*). When the representation has at least as many dimensions as the latent space, we prove that the optimal encoders are *linear* in the latents. Conversely, when dimensions are insufficient, identifiability generically fails and encoders become *nonlinear*. We then show that additional constraints yield recovery of individual latent variables, not just linear mixtures. In the generative setting, non-negative activations and energy efficiency force each neuron to encode exactly one latent, providing a precise account of selective *grandmother cell*-like responses. In the discriminative setting, when latents are sparse and the true measurement matrix satisfies the spark condition, the learned encoder inherits *compressed sensing* structure, enabling sparse dictionary learning to perfectly recover individual latents. This constitutes, to our knowledge, the first identifiability result for undercomplete nonlinear independent component analysis. Our framework unifies nonlinear ICA, compressed sensing, and neural network interpretability, delineating exactly when representations transition from nonlinear distortions to linear representations and, under additional constraints or steps, to clean, axis-aligned codes. Theoretical predictions are validated on synthetic benchmarks and on activations drawn from vision models, large language models, and macaque inferotemporal cortex.

<p align="center">
   <b>[ <a href="paper/paper.pdf">Paper</a> ]</b>
</p>

If you find this work useful, please cite:

```bibtex
@inproceedings{zhou2026complete,
  title={Complete or Sparse: A Tale of Two Identifiabilities},
  author={Zhou, Junze and Klindt, David},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Repository Structure

```
complete-or-sparse/
├── Codes/
│   ├── utils.py                              # Shared numerical functions
│   ├── generate-traces.py                    # Simulation runner for Figs. 1, 5, 6
│   ├── plot-traces.py                        # Plotter for Figs. 1, 5, 6
│   ├── paper-figures.py                      # Paper-sized renderings of Figs. 7, 8
│   ├── fig4_mis.py                           # Minimal MIS implementation for Fig. 4
│   ├── fig1&5&6-axis_aligned_neurons/train.py # Supervised training on the synthetic DGP
│   ├── fig2&9-capacity_phase_transition/     # Thm. 4.3 simulation; Figs. 2, 9
│   ├── fig3-compressed_sensing_recovery/     # Thm. 4.4 simulation; Fig. 3
│   ├── fig4-cross_system_comparison/         # Vision / language / macaque IT; Fig. 4
│   ├── fig7-weight_decay_reliability/        # Weight-decay sweep; Fig. 7
│   ├── fig8-ablation_test/                   # Assumption ablations (config.json); Fig. 8
│   └── UPSTREAM_LICENSE                      # MIT license of adapted upstream code
├── Data/                                     # External-data instructions and generated outputs
├── paper/                                    # Manuscript source, PDF, and archived figures
├── REPRODUCIBILITY.md                        # Data and reproducibility policy
└── requirements.txt
```

Every script reads from and writes to `Data/`, which starts empty apart from [`Data/README.md`](Data/README.md). Generated traces and third-party datasets are never committed. `paper/figures/` holds the archived submission figures and is not an input to any script.

## Installation

```bash
pip install -r requirements.txt
```

Run all commands from the repository root. Defaults reproduce the paper settings (10 seeds per condition). For a fast pipeline check of Figs. 2 and 3, set `COMPLETE_OR_SPARSE_QUICK=1`; quick runs do not reproduce the reported estimates.

## Experiments

### Axis-Aligned Neurons (Figs. 1, 5, 6)

Supervised training on hyperspherical and sparse DGPs with linear, ReLU (Fig. 1), or BN + exponential (Fig. 5) representations. Fig. 6 reuses the Fig. 1/5 sparse runs plus a sparse-ReLU sweep with weight decay 0.03.

```bash
python Codes/generate-traces.py fig1   # also: fig5, fig6 (fig6 requires fig1 and fig5)
python Codes/plot-traces.py fig1       # → Data/fig1/fig1.pdf
```

### Capacity Phase Transition (Figs. 2, 9)

Final KL divergence of linear vs. MLP encoders for `m ∈ {2,4,6,8,10}`, three latent distributions and three teacher scales (Thm. 4.3). Fig. 9 shows the corresponding learning curves.

```bash
python 'Codes/fig2&9-capacity_phase_transition/thm43-simulation.py'
python 'Codes/fig2&9-capacity_phase_transition/fig2.py'   # → Data/fig2/{fig2,fig9}.png
```

### Compressed Sensing Recovery (Fig. 3)

KL / row-space traces and sparse-recovery scores for sparsity `k ∈ {1,2,3}` (Thm. 4.4). Each measurement matrix is exhaustively checked for the spark condition at order `2k`.

```bash
python Codes/fig3-compressed_sensing_recovery/thm44-simulation.py
python Codes/fig3-compressed_sensing_recovery/fig3.py   # → Data/fig3/fig3.png
```

### Cross-System Comparison (Fig. 4)

Unit MIS vs. cross-model MMU in vision models (VGG19-BN, ResNet-50, ViT-B/16, DINOv2-S/14; panels A, C), language models (GPT-2, Pythia-70M, TinyLlama; panels B, D), and macaque IT (panel E). External data are obtained by the user and never redistributed:

| Panels | Data | Source |
|---|---|---|
| A, C | ILSVRC2012 validation set (50,000 images) + DreamSim | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/) |
| B, D | 100,000-word frequency list | [Wiktionary frequency lists](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists) |
| E | 3,200 images × 168 IT units (58 Chabo + 110 Tito) | Brain-Score public [`dicarlo.MajajHong2015.public`](https://github.com/brain-score/vision/tree/master/brainscore_vision/data/majajhong2015); [Majaj et al., 2015](https://doi.org/10.1523/JNEUROSCI.5181-14.2015) |

Panels A–D: set `IMAGENET_VAL_DIR`, `DREAMSIM_EMBEDDINGS`, and `WORD_LIST_PATH`, run `vision.ipynb` and `language.ipynb` from the repository root, then:

```bash
python Codes/fig4-cross_system_comparison/extract-points.py \
  --vision-pdf Data/fig4/raw/vision-scatter-raw.pdf \
  --language-pdf Data/fig4/raw/language-scatter-raw.pdf
python Codes/fig4-cross_system_comparison/panels.py
```

Panel E and the full figure (LPIPS similarity is the longest step; ~9 GB temporary storage, 41 MB final matrix). First manually download the three official Brain-Score files listed in [`Data/README.md`](Data/README.md), then run:

```bash
python Codes/fig4-cross_system_comparison/extract-majaj2015.py \
  --assembly /path/to/assy_dicarlo_MajajHong2015_public.nc \
  --stimulus-csv /path/to/stimulus_hvm-public.csv \
  --stimulus-zip /path/to/stimulus_hvm-public.zip
python Codes/fig4-cross_system_comparison/prepare-majaj2015.py \
  --responses Data/fig4/raw/majaj2015/responses.mat \
  --animals Data/fig4/raw/majaj2015/animals.mat \
  --images-dir Data/fig4/raw/majaj2015/stimuli
python Codes/fig4-cross_system_comparison/fig4.py   # → Data/fig4/fig4.{pdf,png}
```

### Weight-Decay Reliability (Fig. 7)

Thirty zero-weight-decay baseline runs and a paired 10-seed × 4-weight-decay grid on a fixed sparse DGP.

```bash
python Codes/fig7-weight_decay_reliability/fig7.py baseline-sweep --init-seeds 0:10
python Codes/fig7-weight_decay_reliability/fig7.py prepare-dgp \
  --artifact-dir Data/fig7/dgp/sparse --device cpu
python Codes/fig7-weight_decay_reliability/fig7.py sweep \
  --output-root Data/fig7/reliability --dgp-path Data/fig7/dgp/sparse \
  --init-seeds 0:10 --weight-decays 0,0.003,0.01,0.03 --resume
python Codes/fig7-weight_decay_reliability/fig7.py plot   # → Data/fig7/report/fig7.{pdf,png}
```

### Assumption Ablations (Fig. 8)

Nineteen ablation conditions × 10 paired seeds, declared in `config.json`.

```bash
CFG=Codes/fig8-ablation_test/config.json
python Codes/fig8-ablation_test/fig8.py prepare-dgp --plan $CFG --artifact-dir Data/fig8/dgp --device cpu
python Codes/fig8-ablation_test/fig8.py simulate --plan $CFG --dgp-path Data/fig8/dgp \
  --output-root Data/fig8/runs --device cpu --resume
python Codes/fig8-ablation_test/fig8.py aggregate --plan $CFG --results-root Data/fig8/runs \
  --output-dir Data/fig8/report   # → Data/fig8/report/fig8.{pdf,png}
```

## Requirements

- **Python**: `pip install -r requirements.txt` (pinned: `numpy 1.26.4`, `torch 2.5.1`, `scipy 1.13.1`, `matplotlib 3.10.0`, `scikit-learn 1.6.1`)
- **Fig. 4 only**: `torchvision`, `transformers`, `transformer-lens`, `lpips`, `h5py`, `jupyter`

## License

Code adapted from upstream work is released under the MIT License (see [`Codes/UPSTREAM_LICENSE`](Codes/UPSTREAM_LICENSE) and [`Codes/MACHINTERP_LICENSE`](Codes/MACHINTERP_LICENSE)). ImageNet, the word-frequency list, and the Majaj2015 assembly are governed by their own terms.
