import numpy as np
# No other imports allowed!

class LeastSquaresLinearRegressor(object):
    ''' 
    Class providing a linear regression model

    Fit by solving the "least squares" optimization.

    Attributes
    ----------
    * self.w_F : 1D array, size n_features (= F)
        vector of weights for each feature
    * self.b : float
        scalar real bias
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares 

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F : vector of weights for each feature
        * self.b : scalar real bias

        Notes
        -----
        The least-squares optimization problem is:
            \min_{w,b}  \sum_{n=1}^N (y_n - w^T x_n - b)^2
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
