import pandas as pd
import numpy as np
import itertools

# Class to compute best subset of predictors and replicate Figure 3.5 from ESL.
class best_subset:
    
    def __init__(self, X, y):
        # Input predictors and target.
        self.X = X
        self.y = y
        self.p = X.shape[1]

        # Output of all subset models.
        self.models = None
        self.best_models = None


    # Find the best models (minimal RSS)
    def find_best(self):
        best = []
        for i in range(len(self.models)):
            best_of_size = min(self.models[i], key=lambda x: x['rss'])
            for d in self.models[i]:
                d['best'] = 1 if d == best_of_size else 0

    
    # Calculate residual sum of squares (RSS)
    def rss_calc(self, y_hat):
        rss = sum((self.y - y_hat)**2)
        return rss
    

    # Function to get predictions (y_hat)
    def predict(self, X, beta):
        y_hat = X @ beta
        return y_hat

    # Create subsets of predictors.
    def make_subsets(self):
        predictors = self.X.columns
        p = len(predictors)
        all_sets = []
        for j in range(p+1):
            subsets = list(itertools.combinations(predictors, j))
            all_sets.append(subsets)
        return all_sets


    # Function to get OLS estimates.
    def OLS(self, X_subset):
        X_temp = X_subset.copy()
        X_temp.insert(0, 'Intercept', 1)

        X, y = X_temp.to_numpy(), self.y.to_numpy()
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta, X
    

    # Fit function.
    def fit(self):

        # Make predictor subsets.
        subsets = self.make_subsets()

        all_regressions = []
        for i in range(self.p + 1):
            curr_subset_size = len(subsets[i])

            model_storage = []
            for j in range(curr_subset_size):
                predictor_names = list(subsets[i][j])
                X_train = self.X[predictor_names]

                beta, X_w_intercept = self.OLS(X_train)
                y_hat = self.predict(X_w_intercept, beta)
                rss = self.rss_calc(y_hat)
                info = {'variables': predictor_names,
                        'no. vars': i,
                        'betas': beta,
                        'rss': rss}
                model_storage.append(info)
            all_regressions.append(model_storage)
            
        self.models = all_regressions
        self.find_best()
