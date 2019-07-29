import json
import logging
import time
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from base.base_dataset import BaseADDataset
from networks.main import build_network


class OCSVM(object):
    """A class for One-Class SVM models."""

    def __init__(self, kernel='linear', nu=0.1):
        """Init OCSVM instance."""

        self.kernel = kernel
        self.nu = nu
        self.rho = None
        self.gamma = None

        self.model = OneClassSVM(kernel=kernel, nu=nu)
        self.embedding = None

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None
        }

    def set_embedding(self, dataset, embedding_size=100, pretrained_word_vectors=None, embedding_reduction='mean',
                      use_tfidf_weights=False, normalize_embedding=False, device: str = 'cpu'):
        """Sets the word embedding for the text data."""
        self.embedding = build_network('embedding',
                                       dataset,
                                       embedding_size=embedding_size,
                                       pretrained_model=pretrained_word_vectors,
                                       update_embedding=False,
                                       embedding_reduction=embedding_reduction,
                                       use_tfidf_weights=use_tfidf_weights,
                                       normalize_embedding=normalize_embedding)
        self.embedding = self.embedding.to(device)

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Trains the OC-SVM model on the training data."""
        logger = logging.getLogger()

        train_loader, _ = dataset.loaders(batch_size=64, num_workers=n_jobs_dataloader)

        # Training
        logger.info('Starting training...')

        X = ()
        for data in train_loader:
            _, text, _, weights = data
            text, weights = text.to(device), weights.to(device)

            X_batch = self.embedding(text, weights)  # X_batch.shape = (batch_size, embedding_size)
            X += (X_batch.cpu().data.numpy(),)

        X = np.concatenate(X)

        # if rbf-kernel, re-initialize svm with gamma minimizing the numerical error
        if self.kernel == 'rbf':
            self.gamma = 1 / (np.max(pairwise_distances(X)) ** 2)
            self.model = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)

        start_time = time.time()
        self.model.fit(X)
        self.results['train_time'] = time.time() - start_time

        logger.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Tests the OC-SVM model on the test data."""
        logger = logging.getLogger()

        _, test_loader = dataset.loaders(batch_size=64, num_workers=n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')

        idx_label_score = []
        X = ()
        idxs = []
        labels = []
        for data in test_loader:
            idx, text, label_batch, weights = data
            text = text.to(device)
            label_batch = label_batch.to(device)
            weights = weights.to(device)

            X_batch = self.embedding(text, weights)  # X_batch.shape = (batch_size, embedding_size)
            X += (X_batch.cpu().data.numpy(),)
            idxs += idx
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()

        X = np.concatenate(X)

        start_time = time.time()
        scores = (-1.0) * self.model.decision_function(X)
        self.results['test_time'] = time.time() - start_time

        scores = scores.flatten()
        self.rho = -self.model.intercept_[0]

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(idxs, labels, scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test AUC: {:.2f}%'.format(100. * self.results['test_auc']))
        logger.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        logger.info('Finished testing.')

    def save_model(self, export_path):
        """Save OC-SVM model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load OC-SVM model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
