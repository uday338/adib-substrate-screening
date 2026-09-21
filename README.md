# ADIB anode-free substrate screening

Physics-informed computational screening, with ML assistance, of substrate /
anodic current-collector materials for Al deposition in chloroaluminate
aluminium dual-ion batteries (Al plated at the negative electrode, AlCl4-
intercalated into graphite at the positive electrode).

**Status (September 2026): computational screening framework and preliminary
candidate identification.** No experimental or DFT validation of the shortlist
yet. See `docs/report.md`.

## Run everything

```powershell
setx MP_API_KEY "<your Materials Project key>"   # once; open a new terminal
python run_all.py                                  # fetch (cached) -> ML -> screen -> validate -> figures
python run_all.py --offline                        # uses data/ only (for a live demo)
```

Step by step:

| step | command | writes |
|---|---|---|
| 1. MP data (cached) | `python adib_data.py` | `data/screening_summary.csv`, `data/entries/`, `data/thermo/` |
| 2. Literature table | `python literature_benchmark.py` | `data/literature_benchmark.csv` |
| 3. Phase-1 ML benchmark | `python substrate_screening.py --key %MP_API_KEY% --out results/model_comparison.csv --audit-out results/leakage_audit.csv --oof-out results/oof_predictions.csv --data-out results/mp_ml_dataset.csv` | model comparison, leakage audit, OOF predictions, ML dataset |
| 4. Screening | `python adib_screening.py` | `results/screening_candidates.csv`, windows, funnel, sensitivity |
| 5. Validation | `python adib_validate.py` | `results/validation.csv` |
| 5b. Shortlist | `python adib_shortlist.py` | `results/shortlist.csv` |
| 6. ML applicability domain | `python adib_ml_domain.py` | `results/ml_domain*.csv` |
| 7. Figures | `python make_figures.py` | `figures/*.png` |

`python substrate_screening.py --demo` runs the ML code on **synthetic** data as
a unit test only. Its numbers mean nothing about materials.

## Files

| file | what it is |
|---|---|
| `docs/descriptors.md` | pre-registered filters, thresholds and validation rules, with sources |
| `docs/report.md` | research report |
| `adib_data.py` | candidate space (families + rationale) and cached MP retrieval |
| `adib_windows.py` | electrolyte stability window from grand-potential phase diagrams |
| `adib_screening.py` | staged filters, Pareto ranking, sensitivity, confidence |
| `literature_benchmark.py` | literature ground-truth table with verification status |
| `adib_validate.py` | known-material validation (agree / disagree / insufficient) |
| `substrate_screening.py` | Phase-1 ML benchmark: Ridge/RF/XGBoost, leakage audit, LOGO + leave-one-element-out, group bootstrap |
| `adib_ml_domain.py` | external test on an unseen family + applicability domain |
| `adib_shortlist.py` | shortlist table (priority, reasons, confidence, literature status) |
| `make_figures.py` | all figures |
| `adib_workflow.ipynb` | executed notebook of the whole workflow (built by `build_notebook.py`) |
| `archive/` | superseded synthetic-only notebook and CSVs, and early partial real-data logs |

## Knowledge types

Every output keeps three kinds of knowledge apart:
**A, computed** (Materials Project or derived deterministically from MP energies),
**B, literature** (`data/literature_benchmark.csv`, with DOI and verification
status), **C, inferred** (ML predictions, screening tiers).
`results/screening_columns.csv` tags each column.

## Classical ML only

Ridge, RandomForest, XGBoost (scikit-learn, xgboost). No deep learning.
