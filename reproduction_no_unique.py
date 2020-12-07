import os
import errno
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import vectorize
from ppmi import vecs_to_ppmi
from cluster import find_classes

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
]

for d in DATASETS:
    print('processing ' + d + ' dataset')
    data = 'corpora/' + d + '.txt'
    if not os.path.exists(os.path.dirname('exp_output/no_unique/')):
        try:
            os.makedirs(os.path.dirname('exp_output/no_unique/'))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    output = 'exp_output/no_unique/' + d + '.txt'

    """
    1. Vector Embedding
    """
    print('\tvector embedding...')
    vecs, vocab = vectorize(data, unique=False)


    """
    2. Normalization
    """
    print('\tnormalization...')
    ppmi = vecs_to_ppmi(vecs)


    """
    3. PCA and Clustering
    """
    print('\tclustering...', output)
    cls = find_classes(ppmi, vocab, set([tuple(vocab.keys())]), max_k=2, max_pcs=1, scalar=1.3)
    with open(output, 'w') as out:
        for cl in sorted(cls):
            out.write(' '.join(cl) + '\n')
        out.close()
