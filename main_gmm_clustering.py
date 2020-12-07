import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import vectorize
from ppmi import vecs_to_ppmi
from pca import get_pcs
from gmm_cluster import find_classes

DATASETS = [
    'brown',
    # 'english_no_diphthongs',
    # 'english',
    # 'finnish_no_cons',
    # 'finnish',
    # 'french',
    # 'nazarov',
    'parupa',
    'samoan_no_vowels',
    'samoan'
]

for d in DATASETS:
    print('processing ' + d + ' dataset')
    data = 'corpora/' + d + '.txt'
    output = 'output_gmm/' + d + '.txt'

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
    cls = find_classes(ppmi, vocab, set([tuple(vocab.keys())]), max_k=2, max_pcs=1)
    with open(output, 'w') as out:
        for cl in sorted(cls):
            out.write(' '.join(cl) + '\n')
        out.close()
