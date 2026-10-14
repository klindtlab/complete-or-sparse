# Data

This directory contains generated or manually downloaded inputs and is ignored
by Git. Keep `.gitkeep` and this file; all other contents can be rebuilt.

## Figure 4 external data

ImageNet validation images, [DreamSim](https://github.com/ssundaram21/dreamsim)
embeddings, and the 100,000-word frequency list must be obtained or generated
by the user under their original terms. Set
`IMAGENET_VAL_DIR`, `DREAMSIM_EMBEDDINGS`, and `WORD_LIST_PATH` as described in
the repository README.

For the public MajajHong2015 macaque data, manually download these three files
from the official Brain-Score public bucket:

- [`assy_dicarlo_MajajHong2015_public.nc`](https://brainscore-storage.s3.amazonaws.com/brainio-brainscore/assy_dicarlo_MajajHong2015_public.nc), SHA-1 `13d28ca0ce88ee550b54db3004374ae19096e9b9`
- [`stimulus_hvm-public.csv`](https://brainscore-storage.s3.amazonaws.com/brainio-brainscore/stimulus_hvm-public.csv), SHA-1 `5ca7a3da00d8e9c694a9cd725df5ba0ad6d735af`
- [`stimulus_hvm-public.zip`](https://brainscore-storage.s3.amazonaws.com/brainio-brainscore/stimulus_hvm-public.zip), SHA-1 `8aa44e038d7b551efa8077467622f9d48d72e473`

Then run, from the repository root:

```bash
python Codes/fig4-cross_system_comparison/extract-majaj2015.py \
  --assembly /path/to/assy_dicarlo_MajajHong2015_public.nc \
  --stimulus-csv /path/to/stimulus_hvm-public.csv \
  --stimulus-zip /path/to/stimulus_hvm-public.zip
```

The command writes `responses.mat`, `animals.mat`, and 3,200 ordered stimulus
images below `Data/fig4/raw/majaj2015/`. No third-party data are committed.
