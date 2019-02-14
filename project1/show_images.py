'''
Simple script to visualize 28x28 images stored in csv files

Usage
-----
$ python show_images.py --dataset_path data_sandal_vs_sneaker/

'''

import argparse
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

def show_images(X, y):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
    for ii in range(9):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[ii].reshape(28,28), interpolation='nearest', vmin=0, vmax=1, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        cur_ax.set_title('y=%d' % y[ii])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='data_digits_8_vs_9_noisy')
    args = parser.parse_args()

    dataset_path = args.dataset_path

    x_df = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_NF = x_df.values

    y_df = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_N = y_df.values

    show_images(x_NF, y_N)
    plt.show()

