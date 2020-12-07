import numpy as np
from math import exp, log, pi
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from pca import get_pcs

def find_classes(ppmi, vocab, classes = set(), max_k = 3, max_pcs = None):
    _, pcas = get_pcs(ppmi)
    if max_pcs:
        pcas = pcas[:, :max_pcs]
    if pcas.shape[0] == 0 or pcas.shape[0] == 1:
        return classes
    vocab_idx = {v:k for k, v in vocab.items()}
    for i in range(pcas.shape[1]):
        col = pcas[:, i].reshape(-1, 1)
        clusters = cluster(col, max_k, vocab_idx)
        # clusters = [
            # tuple(sorted([vocab_idx[x] for x in np.where(best_kmeans.labels_ == y)[0]])) for y in range(best_kmeans.n_clusters)]
        for c in clusters:
            if c not in classes:
                classes.add(c)
                print('\tnew class: ', c, len(classes))
                if len(c) > 1:
                    c_idx = [vocab[x] for x in c]
                    subset = ppmi[c_idx, :]
                    subvocab = { k: v for v, k in enumerate(sorted(c))}
                    classes.update(find_classes(subset, subvocab, classes))
    return classes

def cluster(pca_col, max_k, vocab_idx):
    gmm = None
    min_bic = -np.inf
    best_clustering = None
    for k in range(1, min(pca_col.shape[0], max_k) + 1):
        gmm = GaussianMixture(n_components = k)
        clustering = gmm.fit_predict(pca_col)
        cl_bic = gmm.bic(pca_col)
        if cl_bic < min_bic or best_clustering is None:
            min_bic = cl_bic
            best_clustering = clustering
    return [
        tuple(sorted([vocab_idx[x] for x in np.where(best_clustering == y)[0]])) for y in range(len(set(best_clustering)))]
