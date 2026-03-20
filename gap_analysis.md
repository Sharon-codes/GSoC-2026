# Gap Analysis: OmniFold HDF5 Weight Files

**Author:** GSoC 2026 Applicant — OmniFold Publication Tools  
**Date:** March 2026  
**Files Analyzed:** `multifold.h5` · `multifold_sherpa.h5` · `multifold_nonDY.h5`  
**Data Source:** [Zenodo record 11507450](https://zenodo.org/records/11507450) — *ATLAS OmniFold 24-Dimensional Z+jets Open Data*

---

## What Are These Files?

Before diving into what's missing, it helps to understand what these files actually *are*. All three are HDF5 files containing Pandas DataFrames, loadable with `pandas.read_hdf()`. They represent the output of running the OmniFold algorithm on ATLAS Z→μμ + jets pseudodata — a 24-dimensional simultaneous unfolding of kinematic observables using a neural network. Instead of giving you a binned histogram of, say, the dimuon transverse momentum, they give you a *weight for every single MC event*, such that when you compute any observable histogram on that weighted sample, you get the particle-level unfolded result.

The three files differ only in what generator or sample composition was used to derive the weights:
- `multifold.h5` — the **nominal** result, using MadGraph5 (MG5)
- `multifold_sherpa.h5` — an **alternative generator** (Sherpa), used as a systematic uncertainty
- `multifold_nonDY.h5` — an **alternative composition** that includes EW Zjj/VBF and diboson on top of the Drell-Yan sample

The idea is that `sherpa` and `nonDY` represent two sources of systematic uncertainty in the unfolding, and you'd estimate their effect by comparing histograms made from each weight set.

---

## 1. What Columns Are Present?

Each file shares the same column schema. The columns fall into two natural groups:

### 1a. Kinematic Observables (24 columns — particle-level truth quantities)

These are frozen at particle level — they represent MC truth values for each event, not reconstructed detector quantities.

| Observable Group | Columns | # | Notes |
|---|---|---|---|
| Leading muon | `mu1_pt`, `mu1_eta`, `mu1_phi` | 3 | Highest-pT muon from Z→μμ |
| Sub-leading muon | `mu2_pt`, `mu2_eta`, `mu2_phi` | 3 | Second muon from Z→μμ |
| Dimuon system | `mumu_pt`, `mumu_y` | 2 | Kinematic quantities of the reconstructed Z |
| Leading jet 4-momentum | `jet1_pt`, `jet1_y`, `jet1_phi`, `jet1_m` | 4 | anti-kT R=0.4 charged-particle jet |
| Sub-leading jet 4-momentum | `jet2_pt`, `jet2_y`, `jet2_phi`, `jet2_m` | 4 | |
| Leading jet substructure | `jet1_nconst`, `jet1_tau1`, `jet1_tau2`, `jet1_tau3` | 4 | N-constituents + N-subjettiness |
| Sub-leading jet substructure | `jet2_nconst`, `jet2_tau1`, `jet2_tau2`, `jet2_tau3` | 4 | |

**Total: 24 observables** — consistent with the Zenodo description of the ATLAS Z+jets 24-dimensional measurement (CERN-EP-2024-132).

### 1b. Weights (the OmniFold output)

Each file contains a column called `weights` (exact name may vary across files as `weights_omnifold` or similar). This single column is the *entire output* of the OmniFold algorithm — a scalar per event. There are no header-level attributes or column-level metadata in the HDF5 file describing what this weight means.

**Classification summary:**
- `mu*`, `jet*`, `mumu_*` columns → **kinematic observables**
- `weights` column → **the OmniFold per-event weight** (the actual deliverable of the analysis)
- No columns that could be clearly classified as standalone **metadata** — identifiers, run numbers, event numbers, etc. are absent entirely

---

## 2. What Information Is Missing for Reuse?

This is the core of the problem. Reading these files without the original analysis documentation is a bit like getting a spreadsheet of numbers with no column headers or context. Here's a detailed breakdown of what a physicist would need that simply isn't there.

### 2.1 What Does This Weight Actually Mean?

The most critical piece of missing information is the **physical interpretation of the weight column**. A weight is just a number — without context, it's meaningless. Specifically, a user needs to know:

- **Is this a Step 1 or Step 2 OmniFold weight?** OmniFold is a two-step iterative procedure. Step 1 reweights detector-level MC to match data. Step 2 pulls those weights back to particle level to produce unfolded results. The file contains the Step 2 (particle-level) weights, but there's no field that says so. If someone mistakenly applied these weights to detector-level quantities, they'd get incorrect results with no warning.

- **What normalization convention is used?** Are the weights normalized so that ∑wᵢ = N (the number of MC events)? Or ∑wᵢ = 1? Or normalized to the measured cross-section in fb⁻¹? This matters enormously when computing cross-sections. Changing the convention silently changes every result by a constant factor.

- **How many OmniFold iterations were run?** The algorithm converges iteratively, and the number of iterations affects the final weights. This is a reproducibility-critical parameter that isn't stored anywhere.

### 2.2 Generator and Dataset Provenance

You can tell from the filenames that `multifold_sherpa.h5` was produced with Sherpa, but that's only because the filename was chosen helpfully. The file itself contains no attribution. Specifically missing:

- Generator name, version, and tune (e.g., "MG5_aMC@NLO v2.9.x, A14 tune")
- Parton distribution function (PDF) set and version (e.g., "NNPDF3.0nlo")
- Monte Carlo cross-section for the simulated sample
- Dataset IDs or ATLAS AMI tags that would let someone trace back exactly which MC production was used
- Pile-up configuration (μ profile, average interactions per bunch crossing)

Without this, two collaborating physicists looking at the same file might disagree on how to normalize their cross-section predictions.

### 2.3 Fiducial Region / Event Selection

This one is really important. The events in the file aren't a random sample — they've been filtered to a specific **particle-level fiducial region**. A user applying these weights to their own events would need to apply the same selection, but there's no description of what that selection is anywhere in the file.

For the ATLAS Z+jets boosted measurement, the fiducial region includes cuts like:
- Dimuon pT > 200 GeV (the "boosted regime" the measurement is designed for)
- Dressed muon pT > 27 GeV and |η| < 2.47
- Z boson mass window [76, 106] GeV

None of this is documented in the HDF5 files. A user working with their own MC would not know whether events with 190 GeV dimuon pT should be included or excluded.

There's also the related question: **what happens to events with zero or one jet?** The file has slots for two jet 4-momenta, but not every event will have two reconstructed jets. Are those events excluded? Are the second-jet columns filled with NaN? With -999? You can't tell from the file alone.

### 2.4 Observable Units and Physical Definitions

The files use column names like `mu1_pt` and `jet1_m`, but nowhere state the units. This sounds like a minor complaint, but ATLAS internally uses MeV for most quantities while published results are in GeV. If a user assumes MeV and the data is actually in GeV, every kinematic cut they apply will be wrong by a factor of 1000.

Beyond units, the jet definition is missing. `jet1_pt` from anti-kT R=0.4 **charged-particle** jets (tracks only, no calorimeter) is a fundamentally different physical quantity from calorimeter jets or particle-level jets using all stable particles. The jet algorithm, radius parameter, and constituent type need to be explicitly documented.

Similarly, the n-subjettiness variables `tau1`, `tau2`, `tau3` depend on a normalization choice (the "beta" parameter) that varies between analyses. Without this, the substructure quantities can't be correctly interpreted or compared.

### 2.5 How the Three Files Relate to Each Other

There is nothing inside any individual file that connects it to the other two. A user who downloads only `multifold_sherpa.h5` has no way of knowing from the file itself that:
1. There is a nominal file they should compare to
2. This file represents a **generator systematic uncertainty** (not an independent measurement)
3. The correct way to use it is to compare weighted histograms to the nominal and take the difference as a systematic band

This relational context — the fact that the three files form a **set** (nominal + two systematics) — is completely absent from the data itself.

### 2.6 How to Use These Weights Correctly

Even for people who understand OmniFold conceptually, there's no specification of the correct workflow:
- Should you weight individual events or sum of weights?
- How do you propagate the systematic envelope? (Min/max of the three weight sets? Quadrature addition?)
- Are the three weight sets applied to the same event sample or different samples?

The answer to the last question is subtle: `multifold.h5` and `multifold_nonDY.h5` use different MC samples entirely (DY-only vs DY+EW+diboson), so you can't directly compare event-by-event. This is non-obvious and isn't documented anywhere.

---

## 3. Challenges in Standardizing This Across Experiments

Even with a perfect metadata schema for this one analysis, making it work universally is genuinely hard for a few reasons I want to think through carefully.

### 3.1 Every Experiment Has Its Own Software World

ATLAS, CMS, LHCb, and Belle II each have decades-old software stacks, internal naming conventions, and data formats. A standard that says "call your weight column `weights_omnifold_step2`" will immediately collide with an experiment that stores weights in columnar ROOT TTrees using `Float_t` arrays, or with a Belle II analysis where the equivalent quantity is called something entirely different. A practical standard needs to either be opinionated (prescribe specific format/schema) or provide a mapping layer — both are hard to get community buy-in for.

### 3.2 The "Systematic" Problem Doesn't Scale

This analysis has two systematic variations. A realistic LHC analysis might have 100–300 systematic variations (jet energy scale, jet energy resolution, luminosity, PDF uncertainty sets, gg→ZZ modelling, ...). If each variation is a separate weight column (or separate file), the metadata schema gets enormous and the files get very large. There's a genuine tension between completeness (store all variations) and usability (don't overwhelm users). The right answer probably involves some kind of tiered system — "core" systematics stored in the main file, "extended" systematics available as optional companion files — but that tiering decision is itself experiment- and analysis-specific.

### 3.3 OmniFold Itself Is Still Evolving

OmniFold as described in the original 2020 paper uses a relatively simple two-step neural network classifier. The community has since developed extensions: MultiFold (which produces weights for detector-level and particle-level simultaneously), flow-based unfolding (which produces a conditional generative model rather than weights), and importance-weighted OmniFold variants. A metadata schema defined today might not cleanly accommodate future methods that produce probability distributions or neural network checkpoints rather than scalar weights.

### 3.4 Data Volume Is a Real Constraint

This Zenodo archive is 3.6 GB for one analysis. Scaled to the LHC Run 4 dataset (orders of magnitude more events), storing per-event weights for public access would be in the terabyte range. HEPData's current infrastructure is not designed for files this large. Any practical standardization effort needs to think about **data reduction strategies** — whether that's compressing weight columns, storing only the ratios to nominal, or defining a lightweight "summary" format for broad reuse alongside the full-resolution archive.

### 3.5 Long-Term Compatibility of HDF5 + Pandas

HDF5 is a mature and robust format, but Pandas' HDF5 interface (via PyTables) has known compatibility issues across library versions. A file written with Pandas 0.25 and `format='fixed'` cannot be read with Pandas 2.0 without manual intervention. If these files are supposed to be usable by physicists 10–20 years from now (as is the stated goal of HEPData archival), the read format, Pandas version minimum, and the HDF5 key structure need to be explicitly documented and tested against future library versions — something no current HEP data archival standard does systematically.

---

## Summary

The three OmniFold HDF5 files are high-quality scientific outputs — they contain exactly what you'd need to reproduce the results of the analysis if you already know what you're doing. But "knowing what you're doing" means having read the paper, worked through the analysis code, and understood the ATLAS software conventions. For community reuse, which is the whole point of publishing OmniFold weights, the files essentially require reading a separate document just to understand the weight column.

The fundamental gap is that **these files are computation artifacts, not published data products**. The transition from one to the other requires adding a standard metadata layer that travels with the files and answers, clearly and completely, the questions: What is this weight? What sample is it for? How do I apply it? What cuts were applied? How does it relate to the other files? Part 2 of this submission proposes a YAML schema to address exactly these questions.
