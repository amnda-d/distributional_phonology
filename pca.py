import numpy as np
from sklearn.decomposition import PCA

def get_pcs(vecs ,scalar=1.0):
    pca = PCA()
    all_pcs = pca.fit_transform(vecs)

    mean_explained_var = np.mean(pca.explained_variance_ratio_)
    top_pcs = all_pcs[:, :max(0, np.argmax(pca.explained_variance_ratio_ < mean_explained_var*scalar))]
    return all_pcs, top_pcs
