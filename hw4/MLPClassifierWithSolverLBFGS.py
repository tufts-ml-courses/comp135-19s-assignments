import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

from scipy.optimize import fmin_l_bfgs_b

import io
import os
import sys
import tempfile
import contextlib

class MLPClassifierLBFGS(MLPClassifier):
    ''' Multi-layer Perceptron classifier which uses L-BFGS to optimize.

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)
    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.
    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    classes_ : array or list of array of shape (n_classes,)
        Class labels for each output.
    loss_ : float
        The current loss computed with the loss function.
    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    n_iter_ : int,
        The number of iterations the solver has ran.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : string
        Name of the output activation function.
    '''


    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation="relu",
            alpha=0.0001,
            max_iter=200,
            tol=1e-4,
            random_state=None,
            ):
        if sklearn.__version__.count('19'):
            version_specific_kws = dict()
        else:
            version_specific_kws = dict(n_iter_no_change=10)
        super(MLPClassifier, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            verbose=True,
            loss='log_loss',  # all opts below here don't matter for LBFGS
            batch_size='auto',
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5, shuffle=True,
            warm_start=False, momentum=0.9,
            nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8,
            **version_specific_kws)

    def fit(self, x, y):
        assert self.solver == 'lbfgs'
        with forcefully_redirect_stdout() as captured_output_dict:
            super(MLPClassifier, self).fit(x, y)
        self.verbose_txt_output = '\n'.join([
            line for line in captured_output_dict['txt'].split('\n')
            if line.strip() != ''])

        self.loss_curve_ = list()
        for line in self.verbose_txt_output.split('\n'):
            if line.startswith('At iterate'):
                loss_str_val = line.split('f=')[1].split('|proj')[0]
                loss_float_val = float(loss_str_val.replace('D', 'e'))
                self.loss_curve_.append(loss_float_val)
        self.loss_curve_ = np.asarray(self.loss_curve_, dtype=np.float64)
        return self


    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(
            self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.loss_, d = fmin_l_bfgs_b(
            x0=packed_coef_inter,
            func=super(MLPClassifier, self)._loss_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads))
        self.optimization_dict = d
        if d['warnflag'] == 0:
            self.did_converge = True
        else:
            self.did_converge = False
        super(MLPClassifier, self)._unpack(optimal_parameters)


def _pack(coefs_, intercepts_):
    """Pack the parameters into a single vector."""
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


@contextlib.contextmanager
def forcefully_redirect_stdout():
    """Redirect stdout at the system level.

    Used to capture data from scipy.optimize.minimize

    Credit: Brandon Dube
    https://gist.github.com/brandondube/2c319727bbc794e97c0806a3abd213ba

    Yields:
        `dict`: dict with a txt key after the context exits
    """
    if type(sys.stdout) is io.TextIOWrapper:
        # Console / command line
        target = sys.stdout
    else:
        # Jupyter
        target = sys.__stdout__

    fd = target.fileno()
    restore_fd = os.dup(fd)
    try:
        tmp, out = tempfile.SpooledTemporaryFile(mode='w+b'), {}
        os.dup2(tmp.fileno(), fd)
        yield out
        os.dup2(restore_fd, fd)
    finally:
        tmp.flush()
        tmp.seek(0)
        out['txt'] = tmp.read().decode('utf-8')
        tmp.close()

if __name__ == '__main__':
    mlp = MLPClassifierLBFGS()
