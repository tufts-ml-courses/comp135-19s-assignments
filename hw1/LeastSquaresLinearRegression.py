import numpy as np


class LeastSquaresLinearRegressor(object):

    def __init__(self):
        ''' Constructor of an sklearn-like regressor
        '''
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares 

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
            \min_{w,b}  \sum_{n=1}^N (y_n - w^Tx_n - b)^2
        '''
        N, F = x_NF.shape
        pass # TODO

    def predict(self, x_NF):
        ''' Make prediction given input features x

        Args
        ----
        x_NF : 2D array, (n_examples, n_features) (N,F)
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_N : 1D array, size N
            Each value is the predicted scalar for one example
        '''
        # TODO FIX ME
        return np.asarray([0.0])

    def print_weights_in_sorted_order(
            self, feat_name_list=None, float_fmt_str='% 7.2f'):
        ''' Print learned coeficients side-by-side with provided feature names

        Args
        ----
        feat_name_list : list of str
            Each entry gives the name of the feature in that column of x_NF
        
        Post condition
        --------------
        Printed all feature names and coef values side by side (one per line)
        Should print all values in w_F first.
        Final line should be the bias value.
        '''
        # TODO

if __name__ == '__main__':
    ## Simple example use case
    # With toy dataset with N=100 examples
    # created via a known linear regression model plus small noise

    prng = np.random.RandomState(0)
    N = 100

    w_F = np.asarray([1.1, -2.2, 3.3])
    x_NF = prng.randn(N, 3)
    y_N = np.dot(x_NF, w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)

    yhat_N = linear_regr.predict(x_NF)
