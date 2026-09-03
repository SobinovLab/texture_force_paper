# Texture force paper

Code to reproduce the figures, tables, and statistics for the tactile
texture-exploration paper "Interaction forces reflect the perception of texture
during active exploration." In the study, 17 participants explored 14 textures
with the index fingertip while normal/tangential force and fingertip kinematics
were recorded, then rated each texture on hardness, slipperiness, and roughness
(H=17, S=16, R=17 participants). The scripts here link interaction forces,
movement, and force vibrations to those perceptual ratings.

## Configuration

All input/output locations live in `config.py` at the repo root; no script
hardcodes paths. It exposes `DATA_DIR`, `OUTPUT_DIR`, `BINNED_FORCE_CSV`,
`BINNED_IMAGE_CSV`, `SESSION_DIR`, and `FRICTION_LONG_TABLE`.

- Set `DATA_DIR` (or the env var `TEXTURE_FORCE_DATA`) to the folder holding:
  - the two binned master CSVs `Subjects1_to_17_Binned_complete_added_columns.csv`
    and `Subjects1_to_17_Binned_Images_Added_Columns.csv`;
  - the raw per-session folders `SubjectN_SessionM/` (with `Force/`, `Images/`,
    `reports.csv`) — needed only for Figures 4/5 and friction extraction;
  - `dynamic_friction_long_table.csv` (a provided input; see friction below).
- Set `OUTPUT_DIR` (or `TEXTURE_FORCE_OUTPUT`) for all generated outputs.
- Defaults are `./data` and `./output`.

Each script's `__main__` prepends the repo root to `sys.path` and imports the
names it needs from `config`, so run scripts from anywhere.

## Layout

```
config.py                 central path configuration
Figure_2/                 Figure 2 (per-subject force medians / CV by task)
Figure_3/                 Figure 3 (texture-level rating vs. predictor grid)
Figure_4/                 Figure 4 (frequency-band vibration vs. rating; band_analysis_cd.py is the pipeline)
Figure_5/                 Figure 5 (friction coefficient vs. rating)
Table_1/                  Table 1 (per-texture force summary stats)
Table_2/                  Table 2 (cross-validated single-predictor regressions)
mixed_model_analysis/     mixed-model analyses:
    Data_Generation_Reliability.py    participant reliability -> Supp Fig 1
    Data_Generation_SuppFig2.py       inter-subject force variance -> Supp Fig 2
    Data_Generation_Band_Trials.py    trial-level band extraction (feeds mixed models)
    Data_Generation_MixedModels.py    subject x texture mixed-effects models
    Data_Generation_Figure_5.py       friction x task interaction control
    revision_utils.py                 shared stats helpers
friction/Multiprocessing.py           dynamic-friction extraction + friction-rating figure
shared/regression_utils.py            shared regression/stats helpers
collect_outputs.py                    zips generated CSV/PNG/SVG/PDF outputs for review
```

## Run order

- For each figure/table, run its `Data_Generation_*.py` first, then its
  `Figure_Generation_*.py` / `Table_Generation_*.py`.
- Figures 4 and 5 and the friction extraction need the raw per-session data;
  Figures 2/3 and Tables 1/2 need only the two binned CSVs.
- Mixed-effects models: run `mixed_model_analysis/Data_Generation_Band_Trials.py`
  (writes the trial-level band table), then
  `mixed_model_analysis/Data_Generation_MixedModels.py`.
- Figures 3/4 and Table 2 exclude two reliability-outlier participant/block pairs
  (`Subject7`/H, `Subject6`/S) by default; Figures 3/4 write the excluded version
  to `*_no_outliers` folders. To keep every participant, set `outlier_pairs = []`
  in those scripts' `__main__`; the mixed-model scripts take
  `--keep_outliers` instead.

## Dependencies

Python 3.11 with: `numpy`, `pandas`, `scipy`, `statsmodels>=0.14`, `matplotlib`,
`seaborn`, `scikit-learn`, `tqdm`. The friction extraction step in
`friction/Multiprocessing.py` additionally needs the lab-internal `reporting_pool`
and `prehension` packages; the friction long table it produces
(`dynamic_friction_long_table.csv`) is otherwise treated as a provided input, so
the rest of the repo runs without those packages.

## Authors

- **Neema Darabi**
- [**Anton Sobinov**](https://github.com/nishbo)

## License

MIT License — see [LICENSE](LICENSE).
