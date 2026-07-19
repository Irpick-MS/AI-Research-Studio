# AI-Research-Studio
AI-assisted scientific communication for independent research. HRSIC Explorer case study.

**Maintainer:** Irpick-MS

*Independent AI-assisted scientific research repository.*

---

## AI-assisted Scientific Communication for Independent Research

This repository contains the statistical validation framework developed for the exploratory **RHSIC (Riemann Harmonic Scale Invariance Conjecture)** case study.

The objective of this project is to provide a fully reproducible Monte Carlo analysis investigating whether the observed correspondence between selected astrophysical systems and the non-trivial zeros of the Riemann zeta function can reasonably arise under different random hypotheses.

**This repository does not claim a proof of the conjecture.**

Instead, it provides:

- Transparent methodology
- Reproducible Python source code
- Datasets
- Monte Carlo simulations
- Statistical reports
- Scientific visualizations

allowing independent verification, discussion, and future improvements.

---

# Repository Structure

```
AI-Research-Studio/

code/
    RHSIC_MonteCarlo_V1_0_100000.py

data/
    rhsic_observed_14_systems.csv
    zeros_5000.txt
    zeros_50000.txt

results/
    monte_carlo_uniform_results.txt
    monte_carlo_permutation_results.txt
    monte_carlo_uniform_100000.csv
    monte_carlo_permutation_100000.csv

figures/
    monte_carlo_uniform_plot.png
    monte_carlo_permutation_plot.png

docs/
    RHSIC_MonteCarlo_Three_Level_Validation.pdf
```

---

# Scientific Background

For each astrophysical system the following quantity is computed

\[
\Phi = \frac{f \times M}{14}
\]

where

- **f** = observed characteristic frequency
- **M** = observed mass
- **14** = normalization factor adopted in the exploratory RHSIC framework.

Each computed value is compared with the nearest non-trivial zero of the Riemann zeta function.

The relative deviation is

\[
\Delta = \frac{|\Phi - t_n|}{t_n}
\]

where \(t_n\) is the nearest non-trivial zeta zero.

---

# Monte Carlo Validation

Version 1.0 performs **100,000 Monte Carlo simulations** using two independent null models.

## Uniform Test

Random frequencies are generated using a logarithmic uniform distribution across the observed range.

This evaluates whether the observed correspondence can emerge purely from randomly generated frequencies.

---

## Permutation Test

Observed frequencies are randomly reassigned to the observed masses.

This preserves the empirical frequency distribution while removing the original frequency-mass associations.

---

# Observed Dataset

The current exploratory dataset contains **14 astrophysical systems**, including

- Stellar black holes
- Intermediate-mass black holes
- Active Galactic Nuclei
- Gravitational-wave ringdown events

---

# Outputs

The software automatically generates

- Monte Carlo reports
- CSV files containing all simulations
- Statistical summaries
- Distribution plots
- Individual deviation analyses

---

# Requirements

Python 3.10 or newer

Required packages

```
numpy
matplotlib
```

Installation

```
pip install numpy matplotlib
```

---

# Running

Execute

```
python RHSIC_MonteCarlo_V1_0_100000.py
```

The program automatically loads the available zeta-zero dataset and performs the complete statistical analysis.

---

# Reproducibility

The simulations are fully reproducible through fixed random seeds.

All parameters used in Version 1.0 are frozen to ensure consistent replication of the published results.

---

# Current Status

**Version 1.0 (Initial Public Release)**

The repository currently includes

- Base Monte Carlo validation
- Uniform null model
- Permutation null model
- Statistical reports
- Figures
- Documentation

Future developments may include

- Leave-One-Out validation
- Threshold sensitivity analysis
- Expanded astrophysical datasets
- Independent replication studies
- Additional statistical robustness tests

---

# AI-assisted Workflow

This project was developed using an AI-assisted scientific workflow.

Large Language Models were used to support

- Python code generation
- Code review
- Documentation
- Scientific illustrations
- Data visualization
- Workflow organization
- Scientific communication

The scientific hypotheses, dataset selection, mathematical formulation, experimental design, interpretation of results, and final editorial decisions remain under the responsibility of the project maintainer.

---

# License

MIT License

---

# Citation

If this repository contributes to your research, please cite it as

**AI Research Studio – RHSIC Monte Carlo Validation Framework**

Version 1.0 (Initial Public Release)

2026

---

# Disclaimer

This repository presents an exploratory scientific investigation intended to encourage independent verification, discussion, and further research.

The statistical analyses, source code, datasets, and documentation are released openly to promote transparency and reproducibility.

No claim of formal mathematical proof is made. Any future scientific conclusions should be based on independent validation, replication, and peer review.

---

*Independent AI-assisted scientific research.*
