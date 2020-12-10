import os
import errno
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import vectorize
from ppmi import vecs_to_ppmi
from cluster import find_classes

DATASETS = [
    'brown', #scalar=1.3
    # 'english', #scalar=1.1
    # 'finnish_no_cons',
    # 'finnish', #scalar=1.2
    # 'french', #scalar=1.7
    # 'nazarov',
    # 'parupa',
    # 'samoan_no_vowels', #scalar=1.3
    # 'samoan'
]

for d in DATASETS:
    print('processing ' + d + ' dataset')
    data = 'corpora/' + d + '.txt'

    if not os.path.exists(os.path.dirname('exp_output/reproduction/')):
        try:
            os.makedirs(os.path.dirname('exp_output/reproduction/'))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    output = 'exp_output/reproduction/' + d + '.txt'

    """
    1. Vector Embedding
    """
    print('\tvector embedding...')
    vecs, vocab = vectorize(data, unique=True)


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
