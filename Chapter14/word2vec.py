from __future__ import print_function

import numpy as np
import multiprocessing

from nltk.corpus import brown
from nltk.corpus import stopwords

# Install Gensim using: pip install -U gensim
# Further information: https://radimrehurek.com/gensim/
from gensim.models import Word2Vec


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    sw = set(stopwords.words('english'))

    # Prepare the corpus
    brown_corpus = brown.sents()

    corpus = []

    for sent in brown_corpus:
        c_sent = [w.strip().lower() for w in sent if w.strip().lower() not in sw]
        corpus.append(c_sent)

    # Train the Word2Vec model
    # A UserWarning: detected Windows; can be discarded
    model = Word2Vec(corpus, size=300, window=10, min_count=1, workers=multiprocessing.cpu_count())
    wv = model.wv
    del model

    # Show a feature vector
    print(wv['committee'])

    print('\n')

    # Show the words most similar to "house"
    print(wv.most_similar('house'))

    print('\n')

    # Show the similarity between "committee" and "president"
    print(wv.similarity('committee', 'president'))