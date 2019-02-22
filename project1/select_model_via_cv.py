'''
Script that evaluates many alpha values using K-fold cross validation.

alpha is the scalar L2 penalty strength within our LR cost function

This script will:
* Load in training data given provided dataset directory (see --dataset_path)
* Loop over the provided alpha values (see --alpha_grid)
* For each alpha, loop over all K folds (see --num_folds)
* Write results to a CSV file (located in folder specified by --results_path)
* Performance metrics at each alpha value are recorded to disk in a CSV file.

This is written as a script that saves intermediate runs to disk.
Thus, you can save time by only calling the alpha values you need.

Notes
-----
We do not randomly shuffle data when assigning the K folds. 

Instead, we assume the data is already stored on disk in random order,
so that each fold consists of a contiguous set of rows.

For example, for 100 examples and 3 folds, we'd assign:
* x_train[0:34] to fold 1/3 (index 0)
* x_train[34:67] to fold 2/3 (index 1)
* x_train[67:100] to fold 3/3 (index 2)

Usage
-----
## Assumes you checked out the entire starter code for project1 to some folder which is your current directory
## That is, you should see the following when you do `ls` or otherwise list your current directory's files and folders:

$ ls
LRGradientDescent.py
select_model_via_cv.py
data_digits_8_vs_9_noisy/
data_sneaker_vs_sandal/

## To do 3-fold CV with alpha 10 and step_size of 1.0
$ python select_model_via_cv.py --dataset_path data_digits_8_vs_9_noisy/ --results_path /tmp/ --alpha_grid 10 --step_size 1.0

## Expected output: alpha0010.0000_cv_scores.csv written to disk inside /tmp/

## To then do 3-fold CV with smaller value of alpha 0.1 then, we could do
$ python select_model_via_cv.py --dataset_path data_digits_8_vs_9_noisy/ --results_path /tmp/ --alpha_grid 0.1 --step_size 1.0

## Expected output: alpha0000.1000_cv_scores.csv written to disk inside /tmp/

We could then compare results saved in CSV files to decide if 0.1 was better than 10
'''
import argparse
import sys
import os
import numpy as np
import pandas as pd
import sklearn.metrics

from LRGradientDescent import LogisticRegressionGradientDescent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
        type=str,
        default='data_digits_8_vs_9_noisy/',
        help='Path to folder where dataset csv files are stored. Must exist already.')
    parser.add_argument('--results_path',
        type=str,
        default='./',
        help='Path to folder where results are saved. Must exist already.')
    parser.add_argument('--alpha_grid', type=str, default='10,100')
    parser.add_argument('--step_size', type=float, default=5.0) # TODO might not be a good value
    parser.add_argument('--num_folds', type=int, default=3)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--init_w_recipe', type=str, default='zeros')
    args = parser.parse_args()

    ## Add parsed args to local namespace
    # So we can reference args like 'dataset_path' directly
    locals().update(args.__dict__)

    if dataset_path is None:
        raise ValueError("Specify a valid --dataset_path that points to folder with x_train.csv inside")
    if results_path is None:
        raise ValueError("Specify a valid --results_path that points to folder to save results inside")

    dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        raise ValueError("Current dataset_path does not exist: %s" % dataset_path)
    results_path = os.path.abspath(results_path)
    if not os.path.exists(results_path):
        raise ValueError("Current results_path does not exist: %s" % results_path)

    try:
        x_NF = np.loadtxt(os.path.join(dataset_path, 'x_train.csv'),
            skiprows=1, delimiter=',')
        y_N = np.loadtxt(os.path.join(dataset_path, 'y_train.csv'),
            skiprows=1, delimiter=',')
    except IOError:
        raise ValueError("Current dataset_path does not have x_train.csv AND y_train.csv inside.")
    ## Determine how to allocate contiguous rows to the K folds
    # Try to have as evenly sized folds as possible
    N = y_N.size
    n_rows_per_fold = int(np.ceil(N / float(num_folds))) * np.ones(num_folds, dtype=np.int32)
    n_surplus = np.sum(n_rows_per_fold) - N
    if n_surplus > 0:
        n_rows_per_fold[-n_surplus:] -= 1
    assert np.allclose(np.sum(n_rows_per_fold), N)
    fold_boundaries = np.hstack([0, np.cumsum(n_rows_per_fold)])
    start_per_fold = fold_boundaries[:-1]
    stop_per_fold = fold_boundaries[1:]

    all_score_dict_list = list()

    for alpha in map(float, alpha_grid.split(',')):
        cur_alpha_score_dict_list = list()

        ## Loop over folds from 1, 2, ... K=num_folds
        for fold_id in range(1, num_folds + 1):
            fold_start = start_per_fold[fold_id-1]
            fold_stop = stop_per_fold[fold_id-1]

            print("alpha %.3g  fold %d/%d of size %5d | validating on rows %5d-%5d of %5d" % (
                alpha, fold_id, num_folds, fold_stop - fold_start, fold_start, fold_stop, N))

            # Training data is everything that's not current validation fold
            x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
            y_tr_N = np.hstack([y_N[:fold_start], y_N[fold_stop:]])

            x_va_NF = x_NF[fold_start:fold_stop].copy()
            y_va_N = y_N[fold_start:fold_stop].copy()

            # Fit the model on current TRAINING split
            lr = LogisticRegressionGradientDescent(
                step_size=step_size, alpha=alpha,
                num_iterations=num_iterations,
                init_w_recipe=init_w_recipe)
            lr.fit(x_tr_NF, y_tr_N)

            # Evaluate on current validation fold
            yproba1_va_N = lr.predict_proba(x_va_NF)[:,1]
            va_score_dict = dict(
                alpha=alpha,
                fold_id=fold_id,
                auroc=sklearn.metrics.roc_auc_score(y_va_N, yproba1_va_N),
                error_rate=sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5),
                log_loss=sklearn.metrics.log_loss(y_va_N, yproba1_va_N),
                did_converge=int(lr.did_converge),
                L1_norm_grad=lr.trace_L1_norm_of_grad[-1],
                step_size=step_size,
                )
            cur_alpha_score_dict_list.append(va_score_dict)
            all_score_dict_list.append(va_score_dict)

            ## Write scores to csv file only for current alpha value
            cur_alpha_cv_scores_df = pd.DataFrame(cur_alpha_score_dict_list)
            cur_alpha_cv_scores_df.to_csv(
                os.path.join(results_path, "alpha%09.4f_cv_scores.csv" % alpha),
                index=False,
                float_format='%.4f',
                columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

            ## Write scores to csv file for ALL alpha values we've tested so far
            all_cv_scores_df = pd.DataFrame(all_score_dict_list)
            all_cv_scores_df.to_csv(
                os.path.join(results_path, "all_cv_scores.csv"),
                index=False,
                float_format='%.4f',
                columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

            ## Write weights to txt file
            model_txt_path = "alpha%09.4f_fold%02d_weights.txt" % (alpha, fold_id)
            lr.write_to_txt_file(os.path.join(results_path, model_txt_path))


