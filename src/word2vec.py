from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    num_docs = len(corpus)
    corpus_vectors = np.zeros((num_docs, num_features))

    for i, doc in enumerate(corpus):
        doc_vector = np.zeros((num_features,))
        word_count = 0
        for word in doc:
            if word in model.wv:
                doc_vector += model.wv[word]
                word_count += 1
        if word_count > 0:
            doc_vector /= word_count
        corpus_vectors[i] = doc_vector

    return corpus_vectors
