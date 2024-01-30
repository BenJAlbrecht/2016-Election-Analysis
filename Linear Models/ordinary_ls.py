import pandas as pd
import numpy as np
import seaborn as sns
import math

from scipy.stats import t

# OLS regression.
class OLS:

    def __init__(self, X, y):
        # Input predictors and target.
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.p = len(self.X.columns)

        # Outputs.
        self.beta = None
        self.y_hat = None
        self.res = None
        self.stderrs = None
        self.zscores = None
        self.pvals = None

        # Summary dataframe.
        self.df = None


    # Function to get OLS estimates.
    def OLS(self):
        X, y = self.X.to_numpy(), self.y.to_numpy()
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y


    # Function to get predictions (y_hat)
    def predict(self):
        X, beta = self.X.to_numpy(), self.beta
        self.y_hat = X @ beta

    
    # Function to get residuals.
    def residuals(self):
        self.res = self.y - self.y_hat

    
    # Function to get s.e. of OLS coefficients.
    def ols_ses(self):
        X = self.X.to_numpy()
        sigma2 = sum(self.res ** 2) / (self.n - self.p)
        mat_xx = np.linalg.inv(X.T @ X)
        stderrs = [math.sqrt(mat_xx[j, j]*sigma2) for j in range(self.p)]
        self.stderrs = stderrs

    
    # Function to get Z-score of OLS coefficients.
    def OLS_z_scores(self):
        scores = self.beta / self.stderrs
        self.zscores = scores


    # Function to get p-vals of OLS coefficients.
    def p_vals(self):
        df = self.n - 1
        p_val = [2 * (1-t.cdf(np.abs(i), df)) for i in self.zscores]
        self.pvals = p_val

    # Function to print summary results.
    def summarize(self):
        df = pd.DataFrame({
            'Term': self.X.columns,
            'Coefficient': np.round(self.beta, 4),
            'Std. Error': np.round(self.stderrs, 3),
            't': np.round(self.zscores, 3),
            'p val.': np.round(self.pvals, 3)
        })
        self.df = df
        print(df)


    # Function to fit model.
    def fit(self):

        # Get OLS estimates.
        self.OLS()

        # Get predicted values.
        self.predict()

        # Get residuals.
        self.residuals()

        # Get standard errors.
        self.ols_ses()

        # Get zscores.
        self.OLS_z_scores()

        # Get p-values.
        self.p_vals()