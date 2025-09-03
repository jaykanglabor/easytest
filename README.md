# Alternative Methods to Test for Positive Assortative Mating
The repo provides a simulated data used for Kang and Stern (2025). 
The replication package is yet incomplete. We are planning to upload the whole replication package.

## Overview

This repository contains MATLAB code to:

1. Construct contingency tables of marital matches from CPS/Census microdata.
2. Fit and simulate an assortative-mating data-generating process (DGP) with diagonal intensity parameter \(\alpha\) and cell-specific shocks.
3. Evaluate four hypothesis tests for Positive Assortative Mating (PAM):
   - **Kang–Stern pseudo-Wald test** (null = random matching)
   - **Kang–Stern likelihood-ratio (LR) test** of independence (null = random matching)
   - **Siow (2015) TP2** and **DP2** restriction-based LR tests (null = PAM), with **parametric bootstrap** critical values and \(p\)-values
4. In particular, **main_section_3.m** includes code generating simulated data  
The code also writes LaTeX tables for inclusion in the manuscript.

There are three empirical/simulation blocks:

- **Census 2000**: fit the DGP, simulate weaker/stronger \(\alpha\), and compare tests.  
- **Same-sex marriage**: compute tests by pooled/male/female samples.  
- **Income-rank (CPS)**: compute tests on wage-rank quintiles (and education).


