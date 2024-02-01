# 2016 Election Analysis
Using linear models to identify the impact of certain state-level features on the 2016 US presidential election. See [Final Analysis](Final%20Analysis.ipynb) for presentation of results and discussion.

For every 1% increase in college attainment of white adults, we expect a 0.4 percentage point increase in the Democratic vote share for a typical state.

These results show the importance of education polarization, especially among white voters in US presidential elections.

## Description
This project uses algorithms I wrote from scratch in Python. 
1) OLS Model. Algorithms:
   * OLS coefficient estimation.
3) Ridge Regression / L2 Regularization. Algorithms used:
   * K-fold cross-validation for hyper-parameter tuning.
   * Newton-Raphson method for identifying problem-dependent hyper-parameters.
   * Ridge regression coefficient estimation.

## GitHub files
* [Data Cleaning Examples](Data%20Cleaning%20Examples)
  * Examples of how I cleaned the sourced data.
* [Linear Models](Linear%20Models)
  * Algorithm implementation Python source files.
     * Ridge Regression (L2): [Implementation](Linear%20Models/ridge.py)
     * Ordinary Least Squares [Implementation](Linear%20Models/ordinary_ls.py)

### Data
Dataset is a compilation of 2016 state-level data from the Federal Election Commission, Census American Community Survey, and website 538.
* FEC data is cleaned to use two-party vote shares.
* ACS data is pulled from ACS 1 year estimates.
* 538 data is average of polls from the final month of election polling.
