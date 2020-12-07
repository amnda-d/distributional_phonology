import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import get_vocab, get_dataset
from ppmi import vecs_to_ppmi
from pca import get_pcs
# from cluster_no_pca import find_classes
from cluster import find_classes
from rnn_embedding import get_rnn_embeddings

DATASETS = [
    # 'brown',
    # 'english_no_diphthongs',
    # 'english',
    # 'finnish_no_cons',
    # 'finnish',
    # 'french',
    # 'nazarov',
    # 'parupa',
    # 'samoan_no_vowels',
    # 'samoan'
    'test'
]

for d in DATASETS:
    print('processing ' + d + ' dataset')
    data = 'corpora/' + d + '.txt'
    output = 'output_context_lstm/' + d + '.txt'
    vecsfile = 'context_lstm_vecs/' + d + '.npy'

    vocab = get_vocab(data)
    dataset = get_dataset(data, unique=False)

    vecs = get_rnn_embeddings(dataset, vocab)#, 2, 20, vecspath = 'context_rnn_vecs/' + d)

    np.save(vecsfile, vecs)

    # vecs = np.load('context_rnn_vecs/brown_1.npy')
    # vecs = vecs_to_ppmi(vecs)

    """
    3. PCA and Clustering
    """
    print('\tclustering...', output)
    cls = find_classes(vecs, vocab, set([tuple(vocab.keys())]), max_k=2, max_pcs=1)
    with open(output, 'w') as out:
        for cl in sorted(cls):
            print(' '.join(cl))
            out.write(' '.join(cl) + '\n')
        out.close()
