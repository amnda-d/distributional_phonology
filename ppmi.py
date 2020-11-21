import numpy as np

def vecs_to_ppmi(vecs):
    total = vecs.sum()
    p_sc = vecs / total
    p_s = vecs.sum(axis=1)[:,np.newaxis] / total
    p_c = vecs.sum(axis=0)[np.newaxis,:] / total

    div = np.divide(p_sc, p_s*p_c, out=np.zeros(p_sc.shape), where=p_s*p_c!=0)
    pmi = np.log2(div, out=np.where(div > 10e-10, div, -1), where=div > 0)
    ppmi = np.where(pmi < 0, 0, pmi)
    return ppmi
