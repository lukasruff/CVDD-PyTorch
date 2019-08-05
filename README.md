# Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text
This repository provides a [PyTorch](https://pytorch.org/) implementation of *Context Vector Data Description (CVDD)*, 
a self-attentive, multi-context one-class classification method for unsupervised anomaly detection on text as presented 
in our ACL 2019 paper.


## Citation and Contact
You find the ACL 2019 paper at [https://www.aclweb.org/anthology/P19-1398](https://www.aclweb.org/anthology/P19-1398).

If you find our work useful, please also cite the paper:
```
@inproceedings{ruff2019,
  title     = {Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text},
  author    = {Ruff, Lukas and Zemlyanskiy, Yury and Vandermeulen, Robert and Schnake, Thomas and Kloft, Marius},
  booktitle = {Proceedings of the 57th Conference of the Association for Computational Linguistics},
  month     = {jul},
  year      = {2019},
  pages     = {4061--4071}
}
```

If you would like to get in touch, just drop an email to [contact@lukasruff.com](mailto:contact@lukasruff.com).


## Abstract
> > There exist few text-specific methods for unsupervised anomaly detection, and for those that do exist, none utilize pre-trained models for distributed vector representations of words. In this paper we introduce a new anomaly detection method---Context Vector Data Description (CVDD)---which builds upon word embedding models to learn multiple sentence representations that capture multiple semantic contexts via the self-attention mechanism. Modeling multiple contexts enables us to perform contextual anomaly detection of sentences and phrases with respect to the multiple themes and concepts present in an unlabeled text corpus. These contexts in combination with the self-attention weights make our method highly interpretable. We demonstrate the effectiveness of CVDD quantitatively as well as qualitatively on the well-known Reuters, 20 Newsgroups, and IMDB Movie Reviews datasets.


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice:
```
git clone https://github.com/lukasruff/CVDD-PyTorch.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-CVDD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-CVDD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

After installing the packages, run `python -m spacy download en` to download the [spaCy](https://spacy.io/) `en` 
library.


## Running experiments
You can run CVDD experiments using the `main.py` script.

The following are examples on how to run experiments on 
[`Reuters-21578`](http://www.daviddlewis.com/resources/testcollections/reuters21578/), 
[`20 Newsgroups`](http://qwone.com/~jason/20Newsgroups/), and 
[`IMDB Movie Reviews`](http://ai.stanford.edu/~amaas/data/sentiment/) as reported in the paper.

### Reuters-21578
Here's an example on `reuters` with `'ship'` (`--normal_class 6`) considered to be the normal class using `GloVe_6B` word
embeddings for a CVDD model with `--n_attention_heads 3` and `--attention_size 150`.
```
cd <path-to-CVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src

# create folder for experimental output
mkdir ../log/test_reuters

# run experiment
python main.py reuters cvdd_Net ../log/test_reuters ../data --device cpu --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 6;
```
The indexation of classes is `[0, 1, 2, 3, 4, 5, 6]` for 
`['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']`.

### 20 Newsgroups
Here's an example on `newsgroups20` with `'comp'` (`--normal_class 0`) considered to be the normal class using 
`FastText_en` word embeddings for a CVDD model with `--n_attention_heads 3` and `--attention_size 150`.
```
cd <path-to-CVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src

# create folder for experimental output
mkdir ../log/test_newsgroups20

# run experiment
python main.py newsgroups20 cvdd_Net ../log/test_newsgroups20 ../data --device cpu --seed 1 --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0;
```
The indexation of classes is `[0, 1, 2, 3, 4, 5]` for `['comp', 'rec', 'sci', 'misc', 'pol', 'rel']`.

### IMDB Movie Reviews
Here's an example on training a CVDD model with `--n_attention_heads 10` and `--attention_size 150` on the full `imdb`
training set (selected via `--normal_class -1`) using `GloVe_42B` word embeddings.
```
cd <path-to-CVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src

# create folder for experimental output
mkdir ../log/test_imdb

# run experiment
python main.py imdb cvdd_Net ../log/test_imdb ../data --device cpu --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_42B --ad_score context_dist_mean --n_attention_heads 10 --attention_size 150 --lambda_p 10.0 --alpha_scheduler soft --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class -1;
```

Have a look into `main.py` for all the possible arguments and options.


## License
MIT
