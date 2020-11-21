import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import vectorize
from ppmi import vecs_to_ppmi
from pca import get_pcs

DATA = '../distributional_learning/corpora/parupa.txt'

"""
1. Vector Embedding
"""
vecs, vocab = vectorize(DATA)


"""
2. Normalization
"""
ppmi = vecs_to_ppmi(vecs)


"""
3. PCA
"""
all_pcs, top_pcs = get_pcs(ppmi)

plt.figure()
for i, target_name in zip(list(vocab.values()), list(vocab.keys())):
    plt.scatter(all_pcs[i, 0], all_pcs[i, 1], alpha=.8,
                label=target_name)
plt.legend(loc='right', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.show()

"""
4. Clustering
"""
