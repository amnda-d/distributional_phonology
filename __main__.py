import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from vectorize import vectorize
from ppmi import vecs_to_ppmi
from pca import get_pcs
from cluster import find_classes

DATA = '../distributional_learning/corpora/parupa.txt'

"""
1. Vector Embedding
"""
vecs, vocab = vectorize(DATA)


"""
2. Normalization
"""
ppmi = vecs_to_ppmi(vecs)
print(ppmi)


"""
3. PCA
"""
all_pcs, top_pcs = get_pcs(ppmi)
print(top_pcs.shape)

# plt.figure()
# for i, target_name in zip(list(vocab.values()), list(vocab.keys())):
#     plt.scatter(all_pcs[i, 0], all_pcs[i, 1], alpha=.8,
#                 label=target_name)
# plt.legend(loc='right', shadow=False, scatterpoints=1)
# plt.title('PCA')
# plt.show()

"""
4. PCA and Clustering
"""
print(vocab)

[print(c) for c in sorted(find_classes(ppmi, vocab, set([tuple(vocab.keys())]), max_k=2, max_pcs=1))]
