# clusterization

import numpy as np
from sklearn.cluster import KMeans
import time


start = time.time()

a = np.array([[0, 0], [10, 0], [5, 5]])
w = np.array([1, 1, 10])
print(a)

result = KMeans(n_clusters=1).fit(a, sample_weight=w)
b = result.cluster_centers_
print(b, b.shape)

print (time.time() - start)