import pandas as pd
import numpy as np

from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate

reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)

## Load the training set into surprise's custom dataset object
train_set = Dataset.load_from_file('data_movie_lens_100k/ratings_train.csv', reader=reader)
train_set = train_set.build_full_trainset()

## Load the test set into surprise's custom dataset object
## (Need to use intermediate pandas DataFrame because the true ratings are missing)
test_df = pd.read_csv('data_movie_lens_100k/ratings_test_masked.csv')
test_set = Dataset.load_from_df(test_df, reader=reader)
test_set = test_set.build_full_trainset().build_testset()

# Use the SVD algorithm
for n_factors in [1]:
    ## Fit model to training set
    model = SVD(n_factors=n_factors)
    model.fit(train_set)

    ## Measure predictions on train set
    tr_pred = model.test(train_set.build_testset())
    tr_mae = accuracy.mae(tr_pred)
    tr_predicted_ratings_N = np.asarray([p.est for p in tr_pred], dtype=np.float64)

    ## Measure predictions on test set
    te_pred = model.test(test_set)
    te_mae = accuracy.mae(te_pred) # should be NaN because no real labels on testset
    te_predicted_ratings_N = np.asarray([p.est for p in te_pred], dtype=np.float64)

    print("n_factors %6d  tr_MAE %7.3f  test_MAE %7.3f" % (n_factors, tr_mae, te_mae))

    print("global mean:")
    print(model.trainset.global_mean)
    print("shape of bias_per_item: ")
    print(model.bi.shape)
    print("shape of bias_per_user: ")
    print(model.bu.shape)
    print("shape of U (per user vectors): ")
    print(model.pu.shape)
    print("shape of V (per item vectors): ")
    print(model.qi.shape)
