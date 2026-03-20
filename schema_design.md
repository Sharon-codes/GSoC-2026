# Designing the Metadata Schema: A Rationale

**Author:** Sharon — GSoC 2026 Applicant  
**Companion File:** `metadata.yaml`  

---

When designing the `metadata.yaml` schema to accompany the OmniFold outputs, my primary goal was to bridge the gap between "data as a computational artifact" and "data as a reusable scientific product." A simple `csv` dump or raw `hdf5` file might be sufficient for the original analyst running the code, but for anyone else—a theorist, an experimentalist on a different dataset, or a phenomenologist ten years from now—it’s essentially unreadable without explicit context.

In this document, I want to walk through the major design decisions I made while structuring the schema, explain exactly *why* I chose certain layouts over others, and discuss the trade-offs involved in standardizing ML-unfolding results for platforms like HEPData.

---

## 1. Why YAML?

Before we get to the content, let’s talk about format. Why YAML and not JSON, XML, or a custom ini file?

1. **Human Readability is Paramount:** While JSON is great for machines, it’s terrible for physicists trying to read documentation quickly. YAML’s indentation-based structure is far more intuitive. 
2. **First-class Support for Comments:** JSON famously doesn't support comments. In scientific metadata, *nuance* is everything. YAML allows me to use inline comments (`#`) to explain *why* a cut is applied, or *what* a particular weight represents, right next to the data.
3. **Multiline Strings:** Describing a complex systematic uncertainty or algorithm often requires a paragraph of text. YAML’s `>` and `|` operators handle multiline prose beautifully.
4. **Community Familiarity:** YAML is increasingly the standard for configuration in modern Python workflows (e.g., Hydra, Snakemake, and even existing HEPData submission formats).

The benefit is a schema that feels more like a structured README than a rigid database entry, yet it can be trivial parsed into a dictionary by any Python script.

---

## 2. Core Structural Choices: The Hierarchy

I organized the schema top-down, mirroring how a physicist approaches a new measurement:

1. **`record` (The "What"):** High-level identity. What paper is this from? What experiment? What is the collision system?
2. **`files` (The "Where" & "How"):** The manifest bridging the physical files to the logical data.
3. **`weights` (The "Core Deliverable"):** The most crucial section, explaining the physical meaning of the numbers inside the files.
4. **`observables` (The "Phase Space"):** Exactly what variables are measured, and in what units.
5. **`fiducial_region` (The "Bounds"):** What is the validity region?
6. **`usage` and `software` (The "Getting Started"):** Code snippets and environment details to actually reproduce the work.

This structure allows automated parsers to easily target specific sections (e.g., a HEPData ingestion script might only care about `observables` and `fiducial_region`), while a user might skip straight to `usage`.

---

## 3. Deep Dive into Specific Design Decisions

### A. The File Manifest: Grouping Nominal and Systematics

One of the most challenging aspects of preserving analysis data is handling systematic uncertainties. In the provided data dump, `multifold_sherpa.h5` and `multifold_nonDY.h5` are just sitting next to `multifold.h5` without any explicit linking. 

I deliberately designed the `files` block not just as a flat list, but as a hierarchical manifest breaking down the `nominal` file vs. a list of `systematics`. 
- **Why?** It enforces a relationship. It explicitly tells the user: *"You use the nominal file for your central value, and here are the specific variations you should use to build your error envelope."* 
- **Granular Generator Tracking:** Notice that I moved the MC generator metadata (like `MG5_aMC@NLO` vs `Sherpa`) into each individual file entry under the manifest. This was critical because the Sherpa systematic variation *literally uses a different generator*. If generator metadata were placed at the top-level `record` block, that vital distinction would be lost.

### B. Weight Specification: Demystifying the Black Box

The `weights` section is the heart of resolving the OmniFold publication problem. To an outside user, an array of numbers could be anything—matrix element weights, pileup weights, or normalization scale factors.

- **`algorithm_step`:** OmniFold is iterative. The distinction between "Step 1" (detector-level reweighting) and "Step 2" (particle-level unfolding) is crucial. By explicitly stating `algorithm_step: "step2"`, we prevent the catastrophic mistake of a user applying these weights to a detector-level simulation.
- **`normalization.convention`:** This is the #1 source of bugs when sharing MC events. I explicitly defined the normalization as `sum_to_N_events`, with a prose description of how to convert that to a physical cross-section. 

### C. Observables and Fiducial Cuts: Leaving Nothing to Guesswork

I listed out all 24 columns in logical groups (Muons, Dimuon System, Jets) rather than an alphabetical flat list. 

- **Units are Mandatory:** Without units, `mu1_pt = 27000` could mean 27 TeV (if mistakenly assumed as GeV) or 27 GeV (if correctly interpreted as MeV). The schema enforces explicitly stated units (`GeV`, `radians`, `dimensionless`).
- **Structured Cuts over Free Text:** Instead of a single string like `mumu_pt > 200 GeV`, I structured cuts into objects (`variable`, `operator`, `value`, `unit`). Why? Because machine-readability matters. A validation script can read this structured format and automatically assert whether an HDF5 DataFrame actually respects the claimed fiducial region using `df[variable] > value`.
- **Handling Jet Algorithms:** A jet is not a fundamental particle; its definition depends entirely on the algorithm. Enforcing the explicit declaration of `jet_algorithm: "anti-kT"`, `jet_radius: 0.4`, and `constituent_type: "charged particles"` ensures the physics isn't lost in translation.

### D. The `usage` Block: Executable Documentation

Including Python code snippets directly inside a metadata YAML is unusual, but I firmly believe it’s the right call here. 

The biggest barrier to entry for using new ML outputs is simply *getting started*. By providing a five-line pandas/numpy snippet that instantly produces a valid weighted histogram, we turn the schema from a passive "dictionary of facts" into an active "quickstart guide." 

---

## 4. Trade-offs, Omissions, and Future Extensions

No schema is perfect, and I had to make deliberate choices to keep this one manageable:

1. **Omission of Training Artifacts:** I did not include the actual multi-layer perceptron (MLP) weights or training logs. While essential for 100% computational reproducibility, they clutter the data-reuse workflow. A physicist using the weights doesn’t need the neural network. I opted for a pointer (`software.omnifold_training`) to describe the architecture. In the future, this could be expanded to link to a separate `Model Card`.
2. **Intermediate Iterations:** OmniFold produces weights at every iteration (e.g., iterations 1 through 5). I only documented the final weights. Storing 5x the data is often prohibitive. However, if a user wants to study convergence, the schema could be extended with an `iterations` array mapping intermediate HDF5 keys.
3. **Statistical Errors & Bootstrapping:** Modern unfolds often use bootstrapping to estimate statistical uncertainties, resulting in hundreds of weight columns (e.g., `weight_boot_001` ... `weight_boot_100`). The current schema doesn't describe large arrays of bootstrap replicas. A future version should add a `statistical_uncertainties` block to map these replica columns efficiently without writing 100 entries.

By striking a balance between rigorous machine-readability and human-friendly design, this schema sets a robust foundation for integrating ML-based unfolding results into repositories like HEPData.
