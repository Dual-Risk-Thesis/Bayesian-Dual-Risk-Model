# Bayesian Estimation of Ruin Probabilities in the Dual Risk Model

This repository contains the R code used for the bachelor’s thesis:

**Bayesian Estimation of Ruin Probabilities in the Dual Risk Model: Accounting for Parameter Uncertainty and Gain-Size Misspecification**

by **Hampus Beijer and Thai Pham**, Linköping University, 2026.

## Overview

The thesis studies Bayesian estimation of the ultimate ruin probability in the classical continuous-time dual risk model. The analysis examines:

- parameter uncertainty under sparse and rich information,
- sensitivity to prior assumptions,
- gain-size model misspecification.

The study is based on simulated data from a compound Poisson dual risk model with exponentially distributed gain sizes.

## Running the analysis
Run the scripts from the repository root directory. The main analysis scripts should be executed in the following order:
```r
source("01_simulation.R")
source("03_run_results_conditional_psi.R")
source("04_prior_sensitivity.R")
source("05_model_misspecification.R")
```

The scripts have the following roles:
1. `01_simulation.R` simulates the sparse-information and rich-information datasets used in the thesis.

2. `03_run_results_conditional_psi.R` runs the baseline Bayesian analysis under the correctly specified exponential gain-size model.

3. `04_prior_sensitivity.R` runs the prior sensitivity analysis using the same simulated datasets.

4. `05_model_misspecification.R` runs the gain-size misspecification analysis using a lognormal gain-size model.

The following files contain helper functions and do not need to be run manually:
* `02_functions_conditional_psi.R`
* `02_functions_lognormal_misspecification.R`

These helper files are sourced automatically by the relevant analysis scripts.

The baseline analysis in `03_run_results_conditional_psi.R` must be completed before running `05_model_misspecification.R`, because the misspecification analysis reads posterior draws produced by the baseline analysis.

The prior sensitivity script can be run after the simulation script and does not require the baseline analysis to be completed first.


## Authors

Hampus Beijer and Thai Pham  
Division of Statistics and Machine Learning  
Linköping University

## License

The R code in this repository is licensed under the MIT License.
