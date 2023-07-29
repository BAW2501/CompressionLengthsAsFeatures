import gzip
import pickle
import time
import numpy as np



with open(f"sentiment-dataset-500.pickle", "rb") as f:
    dataset = pickle.load(f)

train_x, train_y, test_x, test_y = dataset


def compressed_len(s):
    return len(gzip.compress(s.encode()))


def fast_ncd_cdist(x1, x2):
    n = len(x1)
    m = len(x2)
    len_x1sample = [compressed_len(sample) for sample in x1]
    len_x2sample = [compressed_len(sample) for sample in x2]
    dist_mat = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            compressed_len_ij = compressed_len(f'{x1[i]} {x2[j]}')
            min_len = min(len_x1sample[i], len_x2sample[j])
            max_len = max(len_x1sample[i], len_x2sample[j])
            dist_mat[i][j] = (compressed_len_ij - min_len) / max_len
    return dist_mat

start = time.time()
train_ncd = fast_ncd_cdist(train_x, train_x)
test_ncd = fast_ncd_cdist(test_x, train_x)
print(f"Time taken: {time.time() - start}")


