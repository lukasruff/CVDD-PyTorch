# Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text
This repository will provide a [PyTorch](https://pytorch.org/) implementation of 
*Context Vector Data Description (CVDD)*, a self-attentive, multi-context one-class classification method for 
unsupervised anomaly detection on text as presented in our ACL 2019 paper.

**8 Jul 2019: I'm in the process of cleaning up the code. The full code will be released shortly in time for ACL 2019.**


## Citation and Contact
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


## License
MIT
