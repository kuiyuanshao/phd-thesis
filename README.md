# Two-Phase Sampling: Automated Imputation in Analysing Subsamples

Code for a PhD project on two-phase validation designs. A large cohort holds an
error-prone measurement, and a probability subsample is validated against a reference
measurement. The goal is an unbiased regression coefficient with a confidence interval that
covers at the nominal rate, rather than an accurate reconstruction of the missing values. The
project develops three imputation methods for this setting and compares them with design-based
estimators on three simulation designs and one clinical case study.

## Methods

- `SISL/` — Sample Imputation via Sample Likelihood. For designs where selection is informative
  but only the sampling weight is released. Chapter 5.
- `SICG/` — Sample Imputation via Conditional Generative Adversarial Networks. Chapter 6.
- `SIRD/` — Sample Imputation via Conditional Residual Denoising Diffusion Networks. Chapter 7.

## Layout

- `code_infsamp/` — Simulation Design I: informative sampling with suppressed selection variables.
- `code_nutri/` — Simulation Design II: Gaussian and Binomial analysis models.
- `code_surv/` — Simulation Design III: a Cox model.
- `code_vccc/` — the Vanderbilt Comprehensive Care Clinic case study.
- `utils/` — shared R helpers for analysis, tables and figures.
- `benchmarks/` — comparator code (TabCSDI).

The thesis LaTeX project and compiled PDF are stored separately in `../phd-thesis-tex/`.
The main source is `dissertation.tex`; its chapters, figures and bibliography are kept
within that directory.

## Pipeline

Each `code_*` harness uses a numbered-script convention. The exact set differs a little by design,
but the stages are:

- `0x_generate_*` — build the full cohort and draw the phase-two validation samples.
- `03_run_*` — run each imputation method and comparator. The `.R` files run a design locally; the
  `.sl` files submit the same work as Slurm jobs. Comparators include MICE, mixgb, GAIN, TabCSDI and
  generalized raking.
- `04_retrieve_estimates.R` — collect the per-replicate estimates into `results_*.csv`.
- `05_generate_results.R` — write `Summary_Table.csv`.
- `06`–`08` — the thesis figures and tables.

Run each design from its own `code_*` directory. Generated figures stay under
`simulations/results/`; figures used in the thesis are kept separately in
`../phd-thesis-tex/sections/figures/`.

## Data

The raw cohorts and the per-replicate simulation outputs are not in this repository. They are large,
over one terabyte in total, and the clinical data are not ours to release. Only the curated result
files that reproduce the thesis tables and figures are included, mostly under
`*/simulations/results/`, together with the generation seeds and parameters under `*/data/params/`.
Running the pipeline regenerates the rest.

## Requirements

R for the analysis pipeline and Python for the SICG and SIRD networks; see `requirements.txt` for the
Python dependencies. The simulations were run on a Slurm cluster, and the `.sl` scripts hold the job
settings.

## License

MIT. See `LICENSE`.
