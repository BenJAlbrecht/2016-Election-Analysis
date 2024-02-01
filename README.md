# 2016 Election Analysis
Using linear models to identify the impact of certain state-level features on the 2016 US presidential election.

## Description
........

## GitHub files
* [Data Cleaning Examples](Data%20Cleaning%20Examples)
  * Examples of how I cleaned the sourced data.
* [Linear Models](Linear%20Models)
  * Linear models. Analysis refers to Jupyter Notebook with model applied to data, implementation to algorithm in Python source file.
     * Ridge Regression (L2): [Analysis](Linear%20Models/Ridge%20analysis.ipynb) $|$ [Implementation](Linear%20Models/ridge.py)
     * Ordinary Least Squares [Analysis](Linear%20Models/OLS%20analysis.ipynb) $|$ [Implementation](Linear%20Models/ordinary_ls.py)

### Data
Dataset is a compilation of 2016 state-level data from the Federal Election Commission, Census American Community Survey, and website 538.
* FEC data is cleaned to use two-party vote shares.
* ACS data is pulled from ACS 1 year estimates.
* 538 data is average of polls from the final month of election polling.
