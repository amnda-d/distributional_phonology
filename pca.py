import numpy as np
from sklearn.decomposition import PCA

def get_pcs(vecs):
    pca = PCA()
    all_pcs = pca.fit_transform(vecs)

    # mean_explained_var = np.percentile(pca.explained_variance_, 80)
    mean_explained_var = np.mean(pca.explained_variance_ratio_)
    top_pcs = all_pcs[:, :max(0, np.argmax(pca.explained_variance_ratio_ < mean_explained_var*1.0))]
    print(all_pcs.shape, top_pcs.shape)
    return all_pcs, top_pcs
