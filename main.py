import gzip
import pickle
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier

with open(f"sentiment-dataset-500.pickle", "rb") as f:
    train_x, train_y, test_x, test_y  = pickle.load(f)
    
train_x = [s.encode() for s in train_x]
test_x = [s.encode() for s in test_x]

def compressed_len(s):
    return len(gzip.compress(s))

def ncd_cdist_parallel(x1, x2):
    n, m = len(x1), len(x2)
    x = x1 + x2
    len_x = Parallel(n_jobs=-1)(delayed(compressed_len)(s) for s in x) # combined for combined threadpool
    len_x1, len_x2 = len_x[:n], len_x[n:] # for readability can be avoided using n+j 
    def compute_dist(i, j):
        return (compressed_len(x1[i] + b' ' + x2[j]) - min(len_x1[i], len_x2[j])) / max(len_x1[i], len_x2[j])
    
    dist_mat = Parallel(n_jobs=-1)(delayed(compute_dist)(i, j) for i in range(n) for j in range(m))

    return np.reshape(dist_mat, (n, m))

# for _ in range(10):
#     start = time.time()
#     train_ncd = ncd_cdist_parallel(train_x, train_x)
#     test_ncd = ncd_cdist_parallel(test_x, train_x)
#     print("Time:", time.time() - start)
train_ncd = ncd_cdist_parallel(train_x, train_x)
test_ncd = ncd_cdist_parallel(test_x, train_x)
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(train_ncd, train_y)
print("Accuracy:", neigh.score(test_ncd, test_y))
