# OmniFold GSoC 2026 Evaluation Task

**Submitted by:** Sharon | GSoC 2026 Applicant  
**Organisation:** CERN-HSF / ML4Sci  
**Project:** OmniFold Publication Tools  
**Task:** [gsoc2026_evaluation_task.md](https://github.com/wamorkart/omnifold-hepdata/blob/main/gsoc2026_evaluation_task.md)

---

## Overview

This repository contains all required deliverables for the GSoC 2026 evaluation task for the **OmniFold Publication Tools** project. The goal of the project is to standardize how machine-learning-based unfolding results (particularly OmniFold per-event weights from ATLAS measurements) are stored, documented, and shared with the HEP community via HEPData.

The evaluation task uses three HDF5 files from Zenodo ([record 11507450](https://zenodo.org/records/11507450)) containing pre-calculated OmniFold weights for ATLAS Z→μμ + jets pseudodata in a 24-dimensional observable space.

---

## Repository Structure

```
omnifold-gsoc2026/
├── gap_analysis.md          # Part 1: Exploration and gap analysis
├── metadata.yaml            # Part 2: Metadata schema
├── schema_design.md         # Part 2: Justification for schema design
├── weighted_histogram.py    # Part 3: Weighted histogram function + tests
└── README.md                # This file
```

---

## Deliverables

### Part 1 — `gap_analysis.md`

A detailed one-page analysis of the three OmniFold HDF5 files covering:

- All 24 kinematic observable columns (muon pT/η/φ, dimuon system, jet 4-momenta, jet substructure) and the weight column, with classification of each column type.
- **Six categories of missing information** that a physicist would need to reuse the weights:
  1. Weight provenance and normalization convention
  2. OmniFold algorithm step (Step 1 vs Step 2)
  3. Fiducial region / event selection cuts
  4. Observable units and jet algorithm definition
  5. Inter-file relationships (nominal ↔ systematic mapping)
  6. Software version and reproducibility information
- Discussion of **five challenges** in standardizing OmniFold output across experiments: heterogeneous software stacks, varying systematic complexity, evolving ML methods, data volume constraints, and long-term HDF5 compatibility.

---

### Part 2 — `metadata.yaml` + `schema_design.md`

#### `metadata.yaml`

A complete YAML metadata schema structured into sections:

| Section | Content |
|---|---|
| `record` | Experiment, CM energy, paper DOI, Zenodo DOI, schema version |
| `files` | Nominal file + list of systematic variants with generator info |
| `weights` | Algorithm, step, normalization convention, iteration count |
| `observables` | All 24 columns with labels, units, particle-level cuts |
| `fiducial_region` | Structured cut objects (variable, operator, value, unit) |
| `usage` | Inline Python code snippets for loading and histogram |
| `software` | OmniFold version, NN architecture, Pandas HDF5 format |

#### `schema_design.md`

A detailed justification document explaining:
- Why YAML was chosen over JSON
- Why the file manifest uses a nominal + systematics list structure
- Why fiducial cuts are structured objects rather than free-text strings
- Why inline code snippets are included in the YAML
- Known limitations and suggested future extensions

---

### Part 3 — `weighted_histogram.py`

A self-contained Python module containing:

#### `weighted_histogram()` function

```python
result = weighted_histogram(
    observable=obs,       # array-like (N,)
    weights=weights,      # per-event OmniFold weights
    bins=20,
    range=(200, 600),
    density=True,
    observable_label=r"$p_T^{\mu\mu}$ [GeV]",
    plot=True,
    output_path="output.png",
    systematic_weights=[sherpa_weights],
    systematic_labels=["Sherpa"],
)
```

**Key features:**
- Full input validation (shape, finiteness, non-negativity)
- Statistical uncertainty via the **sum-of-squared-weights** estimator (the correct formula for weighted histograms)
- Optional density normalization (integral = 1)
- Optional systematic uncertainty envelope (shaded band)
- Publication-quality Matplotlib figure with error bars
- Returns a structured dict with `bin_edges`, `bin_centers`, `counts`, `stat_errors`, `n_events_per_bin`, `total_weight`

#### Tests

11 pytest tests covering all important edge cases:

| Test | Edge Case | Rationale |
|---|---|---|
| `test_uniform_weights_match_numpy` | w=1 must match `np.histogram` | Basic correctness |
| `test_none_weights_equal_uniform` | `None` ≡ w=1 | API consistency |
| `test_shape_mismatch_raises` | Different array lengths | Catches copy-paste bugs |
| `test_nan_values_excluded` | NaN sentinel values | Common in HEP HDF5 files |
| `test_density_integrates_to_one` | ∫ density dx = 1 | Often broken in naive impl. |
| `test_single_event` | N=1 does not crash | Degenerate input |
| `test_negative_weights_raise` | w < 0 → ValueError | OmniFold invariant |
| `test_all_zero_weights_raise` | Σw = 0 → ValueError | Detect degenerate runs |
| `test_custom_bin_edges` | Non-uniform edges | Log-scale binning use case |
| `test_stat_errors_non_negative` | errors ≥ 0 | Sqrt of sum-of-squares |
| `test_non_1d_input_raises` | 2D input → TypeError | Common user error |

#### Running the tests

```bash
pip install numpy matplotlib pyyaml pytest
pytest weighted_histogram.py -v
```

Expected output:
```
11 passed in 0.21s
```

#### Running the demo

```bash
python weighted_histogram.py
```

This generates `demo_weighted_histogram.png` showing a simulated dimuon pT spectrum with OmniFold weights, statistical error bars, and a systematic uncertainty band.

---

## Background: OmniFold and the Publication Problem

[OmniFold](https://arxiv.org/abs/1911.09107) (Andreassen et al., 2020) is a machine-learning-based unfolding method that produces **per-event weights** rather than binned histograms. This enables flexible reinterpretation — any observable can be computed post-hoc from the stored weights, without rerunning the detector simulation or the neural network.

However, this flexibility comes with a publication challenge: a binned histogram is a self-describing data product (it has axis labels, bin edges, and counts). A per-event weight file is not — it requires extensive metadata to be interpretable by anyone other than the original authors. This GSoC project proposes a standard for that metadata.

---

## References

- OmniFold algorithm: https://arxiv.org/abs/1911.09107
- ATLAS Z+jets measurement: CERN-EP-2024-132
- Zenodo data archive: https://zenodo.org/records/11507450
- OmniFold hepdata project: https://github.com/wamorkart/omnifold-hepdata
- OmniFold Python package: https://github.com/hep-lbdl/OmniFold
- ATLAS public code: https://gitlab.cern.ch/atlas-physics/public/sm-z-jets-omnifold-2024
