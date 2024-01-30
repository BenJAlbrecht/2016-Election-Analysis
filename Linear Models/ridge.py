import pandas as pd
import numpy as np
import scipy.stats
import math

class ridge_reg:

    def __init__(self, X, y, K=10, inc=1, cross_valid=True):
        # Inputs.
        self.X = X      # Predictors for the regression.
        self.y = y      # Target for the regression.
        self.p = X.shape[1] # Predictors.

        # Settings
        self.K = K      # Number of folds to use in K-fold CV.
        self.inc = inc  # Increments for Newton-Raphson method.
        self.cross_valid = cross_valid  # Should we use cross validation.

        # Results.
        self.result = None


    # Convert result into dataframe.
    def result_df(self):
        temp_df = pd.DataFrame(self.result)
        temp_df = temp_df.explode(['edfs', 'rss'])

        temp_df['edfs'] = pd.to_numeric(temp_df['edfs'])
        temp_df['rss'] = pd.to_numeric(temp_df['rss'])

        eDFS = temp_df['edfs'].unique()
        cv_avgs = temp_df.groupby('edfs')['rss'].mean().reset_index(drop=True)

        result_df = pd.DataFrame({
            'edfs': eDFS, 'cv_err': cv_avgs
        })

        std_errs = self.get_std_errs(temp_df, result_df)

        result_df['cv_std_err'] = std_errs

        self.result = result_df


    # Make standard errors from CV_err sample mean.
    def get_std_errs(self, samples, avgs):
        storage_std_errs = []
        for i in range(self.p + 1):
            sample_mask = samples['edfs'] == i
            avgs_mask = avgs['edfs'] == i

            samples_ = samples[sample_mask]['rss']
            n = len(samples_)
            sample_avg = float(avgs[avgs_mask]['cv_err'][i])

            sample_std_dev = math.sqrt(sum((samples_ - sample_avg)**2) / (n-1))
            #std_err = sample_std_dev / math.sqrt(n)

            storage_std_errs.append(sample_std_dev)
        return storage_std_errs



    # Get CV error for edf = 0, the constant model.
    def const_model_result(self, folds):
        rss_, no_, edfs = [], np.nan, np.zeros(self.K)
        for i in range(self.K):
            fold_mask = folds != i
            fold_size = sum(fold_mask == True)

            y_train = self.y[fold_mask]
            y_train_avg = y_train.mean()

            rss = self.rss(y_train, y_train_avg, n=fold_size)
            rss_.append(rss)
        const_model = {'fold_no': no_, 'edfs': edfs, 'rss': rss_}
        return const_model


    # Newton-Raphson for getting EDFs and penalty hyperparameters.
    def newtons_method(self, X_train):
        dfs_no = self.p
        dfs = np.arange(self.inc, dfs_no + self.inc, self.inc)

        U, SIGMA, V = scipy.linalg.svd(X_train, full_matrices=False)
        d = SIGMA**2

        newt_thresh = 1e-3

        reg_params = []
        for df in dfs:
            param_j = 0
            param_j_next = 1e6

            difference = param_j_next - param_j
            while abs(difference) > newt_thresh:
                h_paramj = sum((d)/(d + param_j)) - df
                h_prime_paramj = sum((d)/((d + param_j)**2))
                param_j_next = param_j + (h_paramj) / (h_prime_paramj)
                difference = param_j_next - param_j
                param_j = param_j_next
            reg_params.append(param_j_next)

        return reg_params, dfs


    # Calculate ridge regression coefficient estimates.
    def get_betas(self, X, y, penalty):
        X, y = X.to_numpy(), y.to_numpy()
        I = np.identity(self.p)

        b = np.linalg.inv(X.T @ X + penalty * I) @ X.T @ y
        return b


    # Function to get predictions (y_hat)
    def predict(self, X, beta):
        X = X.to_numpy()
        return X @ beta


    # Function to get CV rss.
    def rss(self, y_tru, y_hat, n):
        return ((y_tru - y_hat)**2).sum() / n


    # Function to get K-folds of data.
    def K_folds(self):
        K, n = self.K, len(self.X)

        idxs_X = np.arange(n)
        idxs_fold = np.zeros(n)

        quo, remain = divmod(n, K)
        fold_size = [quo for _ in range(K)]
        for i in range(remain):
          fold_size[i] += 1

        for j, fold_size in enumerate(fold_size):
          curr_fold = np.random.choice(idxs_X, fold_size, replace=False)
          idxs_fold[curr_fold] = j

          idxs_X = sorted(list(set(idxs_X) - set(curr_fold)))

        return idxs_fold


    # Function to fit model, when we need to use cross-validation.
    def fit_with_cv(self):

        result = []

        K_fold_idxs = self.K_folds()

        const_models = self.const_model_result(K_fold_idxs)
        result.append(const_models)

        # Iterate through folds.
        for i in range(self.K):
            fold_mask = K_fold_idxs != i
            fold_size = sum(fold_mask == True)

            X_train = self.X[fold_mask]
            y_train = self.y[fold_mask]

            penalties, dfs = self.newtons_method(X_train)

            _edfs, _rss = [], []
            for penalty, df in zip(penalties, dfs):
                b = self.get_betas(X_train, y_train, penalty)
                y_hat = self.predict(X_train, b)
                rss = self.rss(y_train, y_hat, n=fold_size)
                _edfs.append(df)
                _rss.append(rss)
            fold_result = {
                'fold_no': i, 'edfs': _edfs, 'rss': _rss
            }
            result.append(fold_result)
        self.result = result
        self.result_df()

    
    # Fit without cross-validation, to examine estimators.
    def fit_no_cv(self):
        penalties, dfs = self.newtons_method(self.X)

        ridge_betas = []
        for l_ in penalties:
            betas = self.get_betas(self.X, self.y, l_)
            ridge_betas.append(betas)
        self.result = ridge_betas    

    # Main fit function.
    def fit(self):
        if self.cross_valid == True:
            self.fit_with_cv()
        elif self.cross_valid == False:
            self.fit_no_cv()