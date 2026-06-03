import numpy as np
import pandas as pd
from GB_generation_with_idx import get_GB
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from scipy.linalg import inv

def MFNE(X, sigma):
    n, m = X.shape
    FNE = np.zeros(m)
    
    remove_x_E = np.zeros((n, m))
    rnc = np.zeros((n, m))
    weight = np.zeros((n, m))
    
    for k in range(m):
        radius = np.std(X[:, k]) / sigma
        temp = 1 - cdist(X[:,[k]],X[:,[k]], metric="mahalanobis")
        temp[temp < radius] = 0
        FNE[k] = -(np.sum(np.log2(np.sum(temp, axis=1) / n))) / n
        weight[:,k] = np.sqrt(np.sum(temp, axis=1) / n)
        for i in range(n):
            temp_x = np.delete(temp, i, axis=0)
            temp_x = np.delete(temp_x, i, axis=1)
            cur_x_n = temp_x.shape[0]
            remove_x_E[i, k] = -(np.sum(np.log2((np.sum(temp_x, axis=1) + 1e-4) / cur_x_n))) / cur_x_n
            rnc[i, k] = np.sum(temp[i]) - (np.sum(temp_x) / cur_x_n)
            
    OD = np.zeros((n, m))
    for i in range(n):
        rne_x = 1 - remove_x_E[i] / FNE
        rne_x[rne_x < 0] = 0
        rne_x[rne_x > 1] = 0    
        for k in range(m):
            if rnc[i, k] > 0:
                OD[i, k] = rne_x[k] * (n - np.abs(rnc[i, k])) / (2 * n)
            else:
                OD[i, k] = rne_x[k] * np.sqrt((n + np.abs(rnc[i, k])) / (2 * n))
    OS = np.zeros(n)
    for i in range(n):
        OS[i] = 1 - (np.sum((1 - OD[i]) * weight[i])) / (m)     
    return OS

if __name__ == '__main__':
    data = pd.read_csv("../datasets/german_1_14_variant1.csv").values
    X = data[:, :-1]
    n, m = X.shape
    labels = data[:, -1]
    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        scaler = MinMaxScaler()
        X[:, ID] = scaler.fit_transform(X[:, ID])

    GBs = get_GB(X)
    n_gb = len(GBs)
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
        
    sigma = 0.4
    OS = MFNE(centers, sigma)
    print(OS)
