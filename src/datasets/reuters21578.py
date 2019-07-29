from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk.corpus import reuters
from nltk import word_tokenize
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import nltk


class Reuters_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']

        # classes_full_list = [
        #     'acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee',
        #     'copper', 'copra-cake', 'corn', 'cotton', 'cotton-oil', 'cpi', 'cpu', 'crude', 'dfl', 'dlr', 'dmk',
        #     'earn', 'fuel', 'gas', 'gnp', 'gold', 'grain', 'groundnut', 'groundnut-oil', 'heat', 'hog', 'housing',
        #     'income', 'instal-debt', 'interest', 'ipi', 'iron-steel', 'jet', 'jobs', 'l-cattle', 'lead', 'lei',
        #     'lin-oil', 'livestock', 'lumber', 'meal-feed', 'money-fx', 'money-supply', 'naphtha', 'nat-gas', 'nickel',
        #     'nkr', 'nzdlr', 'oat', 'oilseed', 'orange', 'palladium', 'palm-oil', 'palmkernel', 'pet-chem', 'platinum',
        #     'potato', 'propane', 'rand', 'rape-oil', 'rapeseed', 'reserves', 'retail', 'rice', 'rubber', 'rye',
        #     'ship', 'silver', 'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'strategic-metal', 'sugar', 'sun-meal',
        #     'sun-oil', 'sunseed', 'tea', 'tin', 'trade', 'veg-oil', 'wheat', 'wpi', 'yen', 'zinc'
        # ]

        self.normal_classes = [classes[normal_class]]
        del classes[normal_class]
        self.outlier_classes = classes

        # Load the reuters dataset
        self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        # Pre-process
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []  # for subsetting train_set to normal class
        for i, row in enumerate(self.train_set):
            if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        test_idx = []  # for subsetting test_set to selected normal and anomalous classes
        for i, row in enumerate(self.test_set):
            if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                test_idx.append(i)
                row['label'] = torch.tensor(0)
            elif any(label in self.outlier_classes for label in row['label']) and (len(row['label']) == 1):
                test_idx.append(i)
                row['label'] = torch.tensor(1)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)
        # Subset test_set to selected normal and anomalous classes
        self.test_set = Subset(self.test_set, test_idx)

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


def reuters_dataset(directory='../data', train=True, test=False, clean_txt=False):
    """
    Load the Reuters-21578 dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)

            examples.append({
                'text': text,
                'label': labels,
            })

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
