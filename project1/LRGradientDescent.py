import numpy as np
from scipy.special import logsumexp

# You can: import any other functions you like from numpy/scipy
# You should not: import from sklearn

class LogisticRegressionGradientDescent():
    ''' Logistic Regression binary classifier trainable via gradient descent.

    Object that implements the standard sklearn binary classifier API:
    * fit : train the model and set internal trainable attributes
    * predict : produce hard binary predictions
    * predict_proba : produce probabilistic predictions for both labels (0 and 1)

    Attributes set by calling __init__()
    ------------------------------------
    alpha : float
    step_size : float
    num_iterations : int

    Attributes set only by calling fit()
    ------------------------------------
    w_G : 1D array, size G
        estimated weight vector
    trace_steps : list
    trace_loss : list
    trace_L1_norm_of_grad : list

    Training Objective
    ------------------
    Find w_G that minimizes calc_loss(w_G, xbias_NG, y_N)

    In math, the loss is defined as:
        L(w) = \frac{1}{N \log 2} (\sum_n log_loss(w, x_n, y_n) + \alpha w^T w)

    We can directly interpret L(w) as an upper bound on the error rate
    on the training data, because:
    * log_loss is an upperbound on zero-one loss when done in base 2
    * the extra L2 penalty term will only ever add to the loss

    Example Usage
    -------------
    >>> x_N1 = np.hstack([np.linspace(-2, -1, 3), np.linspace(1, 2, 3)])[:,np.newaxis]
    >>> y_N = np.hstack([np.zeros(3), np.ones(3)])

    >>> clf = LogisticRegressionGradientDescent(
    ...     alpha=1.0, step_size=0.1, verbose=False)

    ### Shouldn't have any weights if we haven't trained yet
    >>> assert not hasattr(clf, 'w_G')

    ### After training, should have some weights
    >>> clf.fit(x_N1, y_N)
    >>> assert hasattr(clf, 'w_G')

    ### Show the positive-class probability
    >>> proba1_N = clf.predict_proba(x_N1)[:,1]
    >>> print(["%.2f" % phat for phat in proba1_N])
    ['0.08', '0.14', '0.23', '0.77', '0.86', '0.92']

    ### Show the hard binary predictions
    >>> clf.predict(x_N1).tolist()
    [0, 0, 0, 1, 1, 1]
    '''

    def __init__(
            self,
            alpha=1.0,
            step_size=1.0,
            num_iterations=10000,
            loss_converge_thr=0.00001,
            grad_norm_converge_thr=0.001,
            param_converge_thr=0.001,
            verbose=True,
            init_w_recipe='zeros',
            random_state=0,
            proba_to_binary_threshold=0.5,
            ):
        ''' Construct instance and set its attributes

        Args
        ----
        alpha : float
        step_size : float
        num_iterations : int
        loss_converge_thr : float
        grad_norm_converge_thr : float
        verbose : bool

        Returns
        -------
        New instance of this class
        '''
        self.alpha = float(alpha)
        self.num_iterations = int(num_iterations)
        self.step_size = float(step_size)
        self.loss_converge_thr = float(loss_converge_thr)
        self.grad_norm_converge_thr = float(grad_norm_converge_thr)
        self.param_converge_thr = float(param_converge_thr)
        self.verbose = bool(verbose)
        self.proba_to_binary_threshold = float(proba_to_binary_threshold)
        self.init_w_recipe = str(init_w_recipe)
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    ### Load and save to disk

    def write_to_txt_file(self, txtfile):
        np.savetxt(txtfile, self.w_G, fmt='%.18e', delimiter=' ')

    def load_from_txt_file(self, txtfile):
        self.w_G = np.loadtxt(txtfile, delimiter=' ')
    
    ### Prediction API methods

    def predict_proba(self, x_NF):
        ''' Produce soft probabilistic predictions for provided input features

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features_excluding_bias)
            Input features (one row per example).

        Returns
        -------
        yproba_N2 : 2D array, size N x 2
            First column gives probability of zero label (negative)
            Second column gives probability of one label (positive)
            Each entry is a non-negative probability value within (0.0, 1.0)
            Each row sums to one
        '''
        N = x_NF.shape[0]

        ## TODO write code to do prediction for logistic regression!
        # Hint: Be sure to use a numerically stable logistic_sigmoid function

        # TODO replace the placeholder code below
        # Which just predicts 100% probability that class is 0
        yproba0_N1 = np.ones((N, 1)) # <-- TODO replace this line
        yproba1_N1 = np.zeros((N, 1)) # <-- TODO replace this line
        yproba_N2 = np.hstack([yproba0_N1, yproba1_N1])
        return yproba_N2

    def predict(self, x_NF):
        ''' Produce hard binary predictions for provided input features

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features_excluding_bias)
            Input features (one row per example).

        Returns
        -------
        yhat_N : 1D array, size N
            Each entry is a binary value (either 0 or 1)
        '''
        proba_N2 = self.predict_proba(x_NF)
        return np.asarray(
            proba_N2[:,1] >= self.proba_to_binary_threshold,
            dtype=np.int32)

    ### Method for training

    def fit(self, x_NF, y_N):
        ''' Fit logistic regression model to provided training data

        Will minimize the loss function defined by calc_loss

        Returns
        -------
        Nothing. Only internal instance attributes updated.

        Post Condition
        --------------
        Internal attributes are updated:
        * w_G contains the optimal weights
        * trace_loss contains loss at every step of gradient descent
        * trace_L1_norm_of_grad contains L1 norm of grad after every step
        '''
        self.did_diverge = False
        self.did_converge = False
        self.trace_steps = list()
        self.trace_loss = list()
        self.trace_L1_norm_of_grad = list()
        self.trace_w = list()

        ## Setup dimension attributes
        # F : num features excluding bias
        # G : num features including bias 
        self.num_features_excluding_bias = x_NF.shape[1]
        self.F = x_NF.shape[1]
        self.G = self.F + 1

        ## Setup input features with additional 'all ones' column
        xbias_NG = self.insert_final_col_of_all_ones(x_NF)

        ## Initialize w_G according to the selected recipe
        if self.verbose:
            print("Initializing w_G with %d features using recipe: %s" % (
                self.G, self.init_w_recipe))
        w_G = self.initialize_w_G(xbias_NG, y_N)

        ## Run gradient descent!
        # Loop over iterations 0, 1, ..., num_iterations -1, num_iterations
        # We don't do a parameter update on iteration 0, just use the initial w
        if self.verbose:
            print("Running up to %d iters of gradient descent with step_size %.3g" % (
                self.num_iterations, self.step_size))
        for iter_id in range(self.num_iterations + 1):
            if iter_id > 0:
                # TODO update parameter: w_G = ...
                w_G = w_G - 0 # <- TODO replace this line

            loss = self.calc_loss(w_G, xbias_NG, y_N)
            grad_G = self.calc_grad(w_G, xbias_NG, y_N)
            avg_L1_norm_of_grad = np.mean(np.abs(grad_G))

            ## Print information to stdout
            if self.verbose:
                if iter_id < 20 or (iter_id % 20 == 0) or (iter_id % 20 == 1):
                    print('iter %4d/%d  loss % 16.6f  avg_L1_norm_grad % 16.6f  w[0] % 8.3f bias % 8.3f' % (
                        iter_id, self.num_iterations, loss, avg_L1_norm_of_grad,
                        w_G[0], w_G[-1]))

            ## Record information
            self.trace_steps.append(iter_id)
            self.trace_loss.append(loss)
            self.trace_L1_norm_of_grad.append(avg_L1_norm_of_grad)
            self.trace_w.append(w_G)

            ## Assess divergence and raise ValueError as soon as it happens
            self.raise_error_if_diverging(
                self.trace_steps, self.trace_loss, self.trace_L1_norm_of_grad)

            ## Assess convergence and break early if happens
            self.did_converge = self.check_convergence(
                self.trace_steps, self.trace_loss,
                self.trace_L1_norm_of_grad, self.trace_w)
            if self.did_converge:
                break

        if not self.did_diverge:
            self.w_G = w_G
        if self.verbose:
            if self.did_converge:
                print("Done. Converged after %d iterations." % self.trace_steps[-1])
            else:
                print("Done. Did NOT converge.")
        # Done with `fit`.


    ### Methods for gradient descent: calc_loss and calc_grad

    def calc_loss(self, w_G, xbias_NG, y_N):
        ''' Compute total loss for used to train logistic regression.

        Sum of log loss over training examples plus L2 penalty term.

        Args
        ----
        w_G : 1D array, size G 
            Combined vector of weights and bias
        xbias_NG : 2D array, size N x G (n_examples x n_features+1)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        loss : float
            Scalar loss. Lower is better.
        '''
        G = w_G.size
        N = float(y_N.size)
        denom = N * np.log(2)

        ## First term: Calc loss due to L2 penalty on weights
        L2_loss = 0.0 # TODO calc L2 penalty term

        ## Second term: Calc log loss by summing over examples
        log_loss = 0.0 # TODO calc log loss in numerically stable way
        
        ## Add two terms together and return sum divided by N * np.log(2)
        return (L2_loss + log_loss) / denom
        
    def calc_grad(self, w_G, xbias_NG, y_N):
        ''' Compute gradient of total loss for training logistic regression.

        Args
        ----
        w_G : 1D array, size G (G = n_features_including_bias)
            Combined vector of weights and bias
        xbias_NG : 2D array, size N x G (n_examples x n_features_including_bias)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        grad_wrt_w_G : 1D array, size G
            Entry g contains derivative of loss with respect to w_G[g]
        '''
        G = w_G.size
        N = float(y_N.size)
        denom = N * np.log(2)

        # TODO calc gradient of L2 penalty term
        grad_L2_wrt_w_G = np.zeros(G)
        # TODO calc gradient of log loss term
        grad_logloss_wrt_w_G = np.zeros(G)

        return (grad_L2_wrt_w_G + grad_logloss_wrt_w_G) / denom

    ### Helper methods

    def insert_final_col_of_all_ones(self, x_NF):
        ''' Append a column of all ones to provided array.

        Args
        ----
        x_NF : 2D array, size N x F

        Returns
        -------
        xbias_NG : 2D array, size N x G, where G = F+1
            First F columns will be same as input array x_NF
            Final column will be equal to all ones.
        '''
        N = x_NF.shape[0]
        return np.hstack([x_NF, np.ones((N,1))])

    def initialize_w_G(self, xbias_NG, y_N):
        ''' Initialize weight vectors according to this instance's recipe

        Args
        ----
        xbias_NG : 2D array, size N x G (n_examples x n_features_including_bias)
            Input features, with last column of all ones
        y_N : 1D array, size N
            Binary labels for each example (either 0 or 1)

        Returns
        -------
        w_G : 1D array, size G (n_features_including_bias)
            Weight vector, where final entry is the bias
        '''
        F = self.num_features_excluding_bias
        G = F + 1
        if self.init_w_recipe == 'uniform_-1_to_1':
            w_G = self.random_state.uniform(-1, 1, size=G)
        elif self.init_w_recipe == 'zeros':
            w_G = np.zeros(G)
        else:
            raise ValueError("Unrecognized init_w_recipe: %s" % init_w_recipe)
        return w_G

    def check_convergence(self, trace_steps, trace_loss,
                          trace_L1_norm_of_grad, trace_w):
        ''' Assess if current gradient descent run has converged

        We assume that at least 100 iters are needed to verify convergence.
        This might be abundantly cautious, but we'd rather be sure.

        Convergence is assessed on three criteria:
        * loss has stopped changing meaningfully over last 100 iters
            Measured by difference of mean loss from recent iters 100-50 to 50-0.
            Compared against the threshold attribute 'loss_converge_thr'
        * gradient is close enough to zero vector
            Measured by the L1 norm of the gradient vector at latest iteration.
            Compared against the threshold attribute 'grad_norm_converge_thr'
        * weights have not changed significantly over last 100 iters
            Compared against the threshold attribute 'param_converge_thr'

        If all 3 criteria are satisified, we return True.
        Otherwise, we return False.

        Args
        ----
        trace_steps : list of int
            Each entry is an iteration number
            Counts up from 0, 1, 2, ...
        trace_loss : list of scalars
            Each entry is the value of the loss at an iteration.
            Should be generally going down
        trace_L1_norm_of_grad : list of scalars
            Each entry is the L1 gradient norm at an iteration.
            Should be generally going down and approaching zero.
        trace_w : list of 1D arrays

        Returns
        -------
        did_converge : bool
            Boolean flag that indicates if the run has converged.
        '''
        iter_id = trace_steps[-1]
        ## Assess convergence
        if iter_id < 100:
            is_loss_converged = False
            is_grad_converged = False
            is_param_converged = False
        else:
            ## Criteria 1/3: has the loss stopped changing?
            # Calc average loss from 100-50 steps ago
            old_avg_loss = np.mean(trace_loss[-100:-50])
            # Calc average loss from 50-0 steps ago
            new_avg_loss = np.mean(trace_loss[-50:])
            loss_diff = np.abs(old_avg_loss - new_avg_loss)
            is_loss_converged = loss_diff < self.loss_converge_thr

            ## Criteria 2/3: is the gradient close to zero?
            # Check if gradient is small enough
            is_grad_converged = trace_L1_norm_of_grad[-1] < self.grad_norm_converge_thr

            ## Criteria 3/3: have weight vector parameters stopped changing?
            # Check if max L1 diff across all weight values is small enough
            max_param_diff = np.max(np.abs(trace_w[-100] - trace_w[-1]))
            is_param_converged = max_param_diff < self.param_converge_thr
                
        did_converge = is_param_converged and is_loss_converged and is_grad_converged
        return did_converge

    def raise_error_if_diverging(
            self, trace_steps, trace_loss, trace_L1_norm_of_grad):
        ''' Raise error if current gradient descent run is diverging

        Will assess current trace and raise ValueError only if diverging.

        Divergence occurs when:
        * loss is going UP consistently over 10 iterations, when should go DOWN.
        * loss is NaN or infinite
        * any entry of the gradient is NaN or infinite

        Divergence happens in gradient descent when step_size is set too large.
        If divergence is detected, we recommend using a smaller step_size.

        Args
        ----
        trace_steps : list of trace step numbers
            Counts up from 0, 1, 2, ...
        trace_loss : list of loss values
            Should be generally going down
        trace_L1_norm_of_grad : list of L1 gradient norms
            Should be generally going down

        Returns
        -------
        Nothing

        Post Condition
        --------------
        Internal attribute `did_diverge` is set to True or False, as needed. 

        Raises
        ------
        ValueError if divergence is detected.
        '''
        n_completed_iters = len(trace_loss)
        loss = trace_loss[-1]
        L1_norm_grad = trace_L1_norm_of_grad[-1]
        did_diverge = False
        if np.isnan(loss):
            did_diverge = True
            reason_str = 'Loss should never be NaN'            
        elif not np.isfinite(loss):
            did_diverge = True
            reason_str = 'Loss should never be infinite'
        elif np.isnan(L1_norm_grad):
            did_diverge = True
            reason_str = 'Grad should never be NaN'
        elif not np.isfinite(L1_norm_grad):
            did_diverge = True
            reason_str = 'Grad should never be infinite'

        # We need at least 10 completed steps to verify diverging...
        elif n_completed_iters >= 10:
            # Let's look at the 10 most recent steps we took, and compare:
            # * the average loss on steps 10-5
            # * the average loss on steps 5-0
            # If the loss is moving in wrong direction by significant amount,
            # We mark run as diverging and exit early.
            old_loss = np.median(trace_loss[-10:-5])
            new_loss = np.median(trace_loss[-5:])
            perc_change_last10 = (new_loss - old_loss) / (1e-10 + np.abs(old_loss))

            oldnew_loss = np.median(trace_loss[-6:-3])
            newnew_loss = np.median(trace_loss[-3:])
            perc_change_last6 = (newnew_loss - oldnew_loss) / (1e-10 + np.abs(oldnew_loss))

            if perc_change_last10 > 0.50 and perc_change_last6 > 0.50:
                did_diverge = True
                reason_str = 'Loss is increasing but should be decreasing!'

        self.did_diverge = did_diverge
        if did_diverge:
            hint_str = "Try a smaller step_size than current value %.3e" % (
                self.step_size)
            print("ALERT! Divergence detected. %s" % reason_str)
            print("Recent history of loss values:")
            M = np.minimum(10, n_completed_iters)
            for ii in range(M):
                print("iter %4d  loss % 16.6f" % (
                    trace_steps[-M+ii], trace_loss[-M+ii]))
            raise ValueError("Divergence detected. %s. %s." % (
                reason_str, hint_str))


if __name__ == '__main__':
    ## Toy problem 
    #
    # Logistic regression should be able to perfectly predict all 10 examples
    # five examples have x values within (-2, -1) and are labeled 0
    # five examples have x values within (+1, +2) and are labeled 1
    N = 10
    x_NF = np.hstack([np.linspace(-2, -1, 5), np.linspace(1,2, 5)])[:,np.newaxis]
    y_N = np.hstack([np.zeros(5), 1.0 * np.ones(5)])

    lr = LogisticRegressionGradientDescent(
        alpha=0.1, step_size=1.0, init_w_recipe='zeros')

    # Prepare features by inserting column of all 1
    xbias_NG = lr.insert_final_col_of_all_ones(x_NF)

    print("Checking loss and grad at all zeros w vector")
    w_G = np.zeros(2)
    print("w_G = %s" % str(w_G))
    print("loss(w_G) = %.3f" % lr.calc_loss(w_G, xbias_NG, y_N))
    print("grad(w_G) = %s" % str(lr.calc_grad(w_G, xbias_NG, y_N)))

    print("Trying gradient descent")
    lr.fit(x_NF, y_N)
