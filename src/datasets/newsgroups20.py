from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk


class Newsgroups20_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = list(range(6))

        groups = [
            ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
             'comp.windows.x'],
            ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
            ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            ['misc.forsale'],
            ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
            ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
        ]

        self.normal_classes = groups[normal_class]
        self.outlier_classes = []
        del classes[normal_class]
        for i in classes:
            self.outlier_classes += groups[i]

        # Load the 20 Newsgroups dataset
        self.train_set, self.test_set = newsgroups20_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.test_set):
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        for row in datasets_iterator(self.train_set, self.test_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i


def newsgroups20_dataset(directory='../data', train=False, test=False, clean_txt=False):
    """
    Load the 20 Newsgroups dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
        examples = []

        for id in range(len(dataset.data)):
            if clean_txt:
                text = clean_text(dataset.data[id])
            else:
                text = ' '.join(word_tokenize(dataset.data[id]))
            label = dataset.target_names[int(dataset.target[id])]

            if text:
                examples.append({
                    'text': text,
                    'label': label
                })

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
