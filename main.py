import gzip
import pickle
import numpy as np




with open(f"sentiment-dataset-500.pickle", "rb") as f:
    dataset = pickle.load(f)

train_x, train_y, test_x, test_y = dataset #