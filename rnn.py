import os
import errno
import numpy as np
from vectorize import get_vocab, get_dataset
from gmm_cluster import find_classes
from rnn_embedding import get_rnn_embeddings

DATASETS = [
    'brown',
    # 'english_no_diphthongs',
    # 'english',
    # 'finnish_no_cons',
    # 'finnish',
    # 'french',
    # 'nazarov',
    'parupa',
    # 'samoan_no_vowels',
    # 'samoan'
    # 'test'
]

for d in DATASETS:
    print('processing ' + d + ' dataset')
    data = 'corpora/' + d + '.txt'
    if not os.path.exists(os.path.dirname('exp_output/rnn/vecs')):
        try:
            os.makedirs(os.path.dirname('exp_output/rnn/vecs/'))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    output = 'exp_output/rnn/' + d + '.txt'
    vecsfile = 'exp_output/rnn/vecs/' + d + '.npy'
    # output = 'test.txt'
    # vecsfile = 'test.npy'

    vocab = get_vocab(data)
    dataset = get_dataset(data, unique=False)

    vecs = get_rnn_embeddings(dataset, vocab, 2, 20, vecspath = 'exp_output/rnn/vecs/' + d)
    # vecs = get_rnn_embeddings(dataset, vocab, 2, 20, vecspath = 'test/')

    np.save(vecsfile, vecs)

    # vecs = np.load('exp_output/rnn/vecs/brown.npy')

    """
    3. PCA and Clustering
    """
    print('\tclustering...', output)
    cls = find_classes(vecs, vocab, set([tuple(vocab.keys())]), max_k=2, max_pcs=1, scalar=1.0)
    with open(output, 'w') as out:
        for cl in sorted(cls):
            print(' '.join(cl))
            out.write(' '.join(cl) + '\n')
        out.close()
