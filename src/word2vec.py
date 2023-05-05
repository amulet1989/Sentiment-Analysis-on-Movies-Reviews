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
    vectors = []
    for tokens in corpus:
        # Generate a vector for each token and append it to the list of vectors
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        # If there are no word vectors for the tokens, append a zero vector
        if len(word_vectors) == 0:
            vectors.append(np.zeros(model.vector_size))
        else:
            vectors.append(np.mean(word_vectors, axis=0))  # average of the word vectors
    return np.array(vectors)
