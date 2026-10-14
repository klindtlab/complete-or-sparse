# Figure 4

Panels A--D use externally obtained ImageNet and Wikipedia-word inputs. The
`vision.ipynb` and `language.ipynb` notebooks write their intermediate arrays
and vector scatter panels below `Data/fig4/raw`. The vision notebook uses the
MIT-licensed core MIS functions in `Codes/fig4_mis.py`; no neighboring
MachInterp checkout is required. `extract-points.py` converts
those vector panels to compact point tables, and `panels.py` renders A--D.

Panel E is generated from the public Brain-Score assembly. Manually download
the assembly, stimulus CSV, and stimulus ZIP listed in `Data/README.md`, then:

```bash
python Codes/fig4-cross_system_comparison/extract-majaj2015.py \
  --assembly /path/to/assy_dicarlo_MajajHong2015_public.nc \
  --stimulus-csv /path/to/stimulus_hvm-public.csv \
  --stimulus-zip /path/to/stimulus_hvm-public.zip
python Codes/fig4-cross_system_comparison/prepare-majaj2015.py \
  --responses Data/fig4/raw/majaj2015/responses.mat \
  --animals Data/fig4/raw/majaj2015/animals.mat \
  --images-dir Data/fig4/raw/majaj2015/stimuli
python Codes/fig4-cross_system_comparison/fig4.py
```

`fig4.py` computes MIS over `k = 2, 4, 8, 16, 32`, computes MMU across the two
animals, renders panel E, and composes the five generated panel PDFs. All
outputs stay below `Data/fig4`; no macaque recording, image, or precomputed
similarity matrix is committed.
