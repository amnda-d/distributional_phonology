import numpy as np
from sklearn.decomposition import PCA

def get_pcs(vecs):
    pca = PCA()
    all_pcs = pca.fit_transform(vecs)

    mean_explained_var = np.mean(pca.explained_variance_)

    top_pcs = all_pcs[:, :max(0, np.argmax(pca.explained_variance_ < mean_explained_var))]

    return all_pcs, top_pcs
