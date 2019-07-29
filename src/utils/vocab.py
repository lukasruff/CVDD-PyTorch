class Vocab(object):
    """ A class to hold and build vocabularies. """

    def __init__(self, words=None):
        super().__init__()

        self.itos = []
        self.stoi = {}
        self.counts = {}

        if words is not None:
            self.add_words(words)

    def add_words(self, words):
        """ Add words (list of str or str) to vocabulary """

        if isinstance(words, list):
            for word in words:
                if isinstance(word, str):
                    if word in self.itos:
                        self.counts[word] += 1
                    else:
                        self.itos.append(word)
                        self.stoi[word] = len(self.itos)
                        self.counts[word] = 1
                else:
                    raise TypeError("words in list must be str")

        elif isinstance(words, str):
            if words in self.itos:
                self.counts[words] += 1
            else:
                self.itos.append(words)
                self.stoi[words] = len(self.itos)
                self.counts[words] = 1
        else:
            raise TypeError("words must be list of str or str")

    def top_words(self, k):
        """ Returns list of (<word>, <count>) pairs of top-k words in vocabulary by occurrence counts. """
        return sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
