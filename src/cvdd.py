from base.base_dataset import BaseADDataset
from networks.main import build_network
from optim.cvdd_trainer import CVDDTrainer

import json


class CVDD(object):
    """A class for Context Vector Data Description (CVDD) models."""

    def __init__(self, ad_score='context_dist_mean'):
        """Init CVDD instance."""

        # Anomaly score function
        self.ad_score = ad_score

        # CVDD network: pretrained_model (word embedding or language model) + self-attention module + context vectors
        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.train_dists = None
        self.train_top_words = None

        self.test_dists = None
        self.test_top_words = None
        self.test_att_weights = None

        self.results = {
            'context_vectors': None,
            'train_time': None,
            'train_att_matrix': None,
            'test_time': None,
            'test_att_matrix': None,
            'test_auc': None,
            'test_scores': None
        }

    def set_network(self, net_name, dataset, pretrained_model, embedding_size=None, attention_size=150,
                    n_attention_heads=3):
        """Builds the CVDD network composed of a pretrained_model, the self-attention module, and context vectors."""
        self.net_name = net_name
        self.net = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model,
                                 update_embedding=False, attention_size=attention_size,
                                 n_attention_heads=n_attention_heads)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 25,
              lr_milestones: tuple = (), batch_size: int = 64, lambda_p: float = 1.0,
              alpha_scheduler: str = 'logarithmic', weight_decay: float = 0.5e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the CVDD model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = CVDDTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, lambda_p, alpha_scheduler,
                                   weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

        # Get results
        self.train_dists = self.trainer.train_dists
        self.train_top_words = self.trainer.train_top_words
        self.results['context_vectors'] = self.trainer.c
        self.results['train_time'] = self.trainer.train_time
        self.results['train_att_matrix'] = self.trainer.train_att_matrix

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the CVDD model on the test data."""

        if self.trainer is None:
            self.trainer = CVDDTrainer(device, n_jobs_dataloader)

        self.trainer.test(dataset, self.net, ad_score=self.ad_score)

        # Get results
        self.test_dists = self.trainer.test_dists
        self.test_top_words = self.trainer.test_top_words
        self.test_att_weights = self.trainer.test_att_weights
        self.results['test_time'] = self.trainer.test_time
        self.results['test_att_matrix'] = self.trainer.test_att_matrix
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_path):
        """Save CVDD model to export_path."""
        # TODO: Implement save_model
        pass

    def load_model(self, import_path, device: str = 'cuda'):
        """Load CVDD model from import_path."""
        # TODO: Implement load_model
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
