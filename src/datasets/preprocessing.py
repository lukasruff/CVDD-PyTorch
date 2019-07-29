from sklearn.feature_extraction.text import TfidfTransformer

import torch
import numpy as np


def compute_tfidf_weights(train_set, test_set, vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from train_set)."""

    transformer = TfidfTransformer()

    # fit idf vector on train set
    counts = np.zeros((len(train_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(train_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(train_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    # compute tf-idf weights for test set (using idf vector from train set)
    counts = np.zeros((len(test_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(test_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.transform(counts)

    for i, row in enumerate(test_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())
