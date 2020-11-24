import codecs
from collections import defaultdict
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
    return list(set(dataset))

def get_trigram_contexts(vocab):
    """
    Dictionary of trigram token contexts mapped to indices: { X_X: i}
    """
    # return { k: v for v, k in enumerate([c1 + '_' + c2 for c1 in list(vocab.keys()) + ['#'] for c2 in list(vocab.keys()) + ['#']])}
    v = list(vocab.keys()) + ['#']
    return { k: v for v, k in enumerate(
        [c1 + '_' + c2 for c1 in v for c2 in v] +
        ['_' + c1 + c2 for c1 in v for c2 in v] +
        [c1 + c2 + '_' for c1 in v for c2 in v]
    )}

def get_trigram_count_vectors(vocab, contexts, dataset):
    vecs = np.zeros((len(vocab), len(contexts)))
    ngrams = {v: defaultdict(int) for v in vocab}
    for w in dataset:
        if w != '':
            w = ['#']*2 + w.split(' ') + ['#']*2
            for i in range(2, len(w) - 2):
                # target = vocab[w[i]]
                target = w[i]
                # if i == 2:
                #     # context = contexts['_' + w[i+1] + w[i+2]]
                #     context = '_' + w[i+1] + w[i+2]
                # elif i == (len(w) - 3):
                #     # context = contexts[w[i-2] + w[i-1] + '_']
                #     context = w[i-2] + w[i-1] + '_'
                # else:
                #     # context = contexts[w[i-1] + '_' + w[i+1]]
                #     context = w[i-1] + '_' + w[i+1]
                # vecs[target][context] += 1

                ngrams[target]['_' + w[i+1] + w[i+2]] += 1
                ngrams[target][w[i-2] + w[i-1] + '_'] += 1
                ngrams[target][w[i-1] + '_' + w[i+1]] += 1
    all_ngrams = [list(ngrams[v].keys()) for v in ngrams.keys()]
    all_ngrams = set([x for l in all_ngrams for x in l])
    vecs = np.zeros((len(vocab), len(all_ngrams)))
    print(ngrams)
    for v, x in enumerate(vocab):
        for i, c in enumerate(all_ngrams):
            vecs[v, i] = ngrams[x][c]
    return vecs

def vectorize(data):
    vocab = get_vocab(data)
    dataset = get_dataset(data)
    contexts = get_trigram_contexts(vocab)
    return get_trigram_count_vectors(vocab, contexts, dataset), vocab
