# Research Project CSE3000 2022

This repository links to the work of students for the Research project course of the CSE bachelor at TU Delft.

Please see their projects [here](https://cse3000-research-project.github.io/).

## An empirical study of the effects of unconfoundedness on the performance of Propensity Score Matching

This codebase has been used to generate the results seen in the paper related to this specific project.
All the appropriate functions can be found in main.py.

### Experiment 1: Effect of hiding individual confounding and non-confounding features

The methods experiment_common_missing_variables_ATE and experiment_common_missing_variables_ATE_LR in main.py have been utilized to obtain the results seen in the paper. The first one uses Propensity Score Matching and the second Linear Regression.

### Experiment 2: Effect of hiding individual features with different effect contributions

The method evh_final in main.py has been utilized to obtain the results seen in the paper. The boolean lr is used to change between using Propensity Score Matching (=False) and Linear Regression (=True).

### Experiment 3: Effect of hiding multiple sets of features on synthetic datasets

The method enhv_final in main.py has been utilized to obtain the results seen in the paper. The boolean lr is used to change between using Propensity Score Matching (=False) and Linear Regression (=True).