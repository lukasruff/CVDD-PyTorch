from torchnlp.word_to_vector import GloVe, FastText


def load_word_vectors(word_vectors_name, embedding_size, word_vectors_cache='../data/word_vectors_cache'):

    implemented_vector_embeddings = ('GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en')
    assert word_vectors_name in implemented_vector_embeddings

    word_vectors = None

    if word_vectors_name == 'GloVe_6B':
        assert embedding_size in (50, 100, 200, 300)
        word_vectors = GloVe(name='6B', dim=embedding_size, cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_42B':
        embedding_size = 300
        word_vectors = GloVe(name='42B', cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_840B':
        embedding_size = 300
        word_vectors = GloVe(name='840B', cache=word_vectors_cache)

    if word_vectors_name == 'GloVe_twitter.27B':
        assert embedding_size in (25, 50, 100, 200)
        word_vectors = GloVe(name='twitter.27B', dim=embedding_size, cache=word_vectors_cache)

    if word_vectors_name == 'FastText_en':
        embedding_size = 300
        word_vectors = FastText(language='en', cache=word_vectors_cache)

    return word_vectors, embedding_size
