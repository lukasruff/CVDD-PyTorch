from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors

import torch


class TorchnlpDataset(BaseADDataset):
    """TorchnlpDataset class for datasets already implemented in torchnlp.datasets."""

    def __init__(self, root: str):
        super().__init__(root)
        self.encoder = None  # encoder of class Encoder() from torchnlp

    def loaders(self, batch_size: int, shuffle_train=False, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):

        # Use BucketSampler for sampling
        train_sampler = BucketBatchSampler(self.train_set, batch_size=batch_size, drop_last=True,
                                           sort_key=lambda r: len(r['text']))
        test_sampler = BucketBatchSampler(self.test_set, batch_size=batch_size, drop_last=True,
                                          sort_key=lambda r: len(r['text']))

        train_loader = DataLoader(dataset=self.train_set, batch_sampler=train_sampler, collate_fn=collate_fn,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_sampler=test_sampler, collate_fn=collate_fn,
                                 num_workers=num_workers)
        return train_loader, test_loader


def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    indices = [row['index'] for row in batch]
    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])
    weights = [row['weight'] for row in batch]
    # check if weights are empty
    if weights[0].nelement() == 0:
        weight_batch = torch.empty(0)
    else:
        weight_batch, _ = stack_and_pad_tensors([row['weight'] for row in batch])
        weight_batch = transpose(weight_batch)

    return indices, transpose(text_batch), label_batch.float(), weight_batch
