import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk

def get_vocab(file):
    """
    Dictionary of unique tokens (phonemes) mapped to indices.
    """
    with codecs.open(file, encoding='utf-8') as f:
        return { k: v for v, k in enumerate(sorted(set(f.read().split())))}

def get_dataset(file):
    """
    Get words from file, one per line.
    """
    with codecs.open(file, encoding='utf-8') as f:
        dataset = []
        for line in f.readlines():
            dataset += [line.strip('\n')]
    return dataset

def get_trigram_contexts(vocab):
    """
    Dictionary of trigram token contexts mapped to indices: { X_X: i}
    """
    return { k: v for v, k in enumerate([c1 + '_' + c2 for c1 in list(vocab.keys()) + ['#'] for c2 in list(vocab.keys()) + ['#']])}

def get_trigram_count_vectors(vocab, contexts, dataset):
    vecs = np.zeros((len(vocab), len(contexts)))
    for w in dataset[3:]:
        if w != '':
            padded = ['#']*2 + w.split(' ') + ['#']*2
            for tri in range(len(padded) - 2):
                trigram = padded[tri: tri+3]
                if trigram[1] != '#':
                    target = vocab[trigram[1]]
                    context = contexts[trigram[0] + '_' + trigram[2]]
                    vecs[target][context] += 1
    return vecs

def vectorize(data):
    vocab = get_vocab(data)
    dataset = get_dataset(data)
    contexts = get_trigram_contexts(vocab)
    return get_trigram_count_vectors(vocab, contexts, dataset), vocab
