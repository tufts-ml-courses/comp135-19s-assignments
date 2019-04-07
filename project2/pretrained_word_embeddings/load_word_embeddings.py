import pandas as pd
import numpy as np
import sklearn.neighbors

from collections import OrderedDict

word_embeddings = pd.read_csv('glove.6B.50d.txt.zip',
                               header=None, sep=' ', index_col=0,
                               nrows=100000, compression='zip', encoding='utf-8', quoting=3)
# Build a dict that will map from string word to 50-dim vector
word_list = word_embeddings.index.values.tolist()
word2vec = OrderedDict(zip(word_list, word_embeddings.values))

## Show some examples
n_words = len(word2vec.keys())
print("word2vec['london'] = ")
print(word2vec['london'])

print("word2vec['england'] = ")
print(word2vec['england'])

## Try some analogies
def analogy_lookup(a1, a2, b1):
    target_vec = word2vec[a2] - word2vec[a1] + word2vec[b1]
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=7, metric='euclidean', algorithm='brute')
    knn.fit(word_embeddings.values)
    dists, indices = knn.kneighbors(target_vec[np.newaxis,:])
    print("Query: %s:%s -> %s:____" % (a1, a2, b1))
    for ii, vv in enumerate(indices[0]):
        print("   %20s  at dist %.3f" % (word_list[vv], dists[0,ii]))

analogy_lookup('england', 'london', 'france')
analogy_lookup('england', 'london', 'germany')
analogy_lookup('england', 'london', 'japan')
analogy_lookup('england', 'london', 'indonesia')

analogy_lookup('swim', 'swimming', 'run')
