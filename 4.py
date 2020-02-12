# reading geotiff

import numpy as np
import json
import rasterio
import time
from sklearn.cluster import KMeans
from skimage.measure import block_reduce

start = time.time()

year = 2000
size = 10 ** 5
blockSize = 3

image_path = './data/mda_ppp_' + str(year) + '.tif'
data_set = rasterio.open(image_path)

left, bottom, right, top = data_set.bounds
w, h = data_set.width // blockSize, data_set.height // blockSize
step = (right - left) * blockSize / data_set.width

a = data_set.read().reshape((data_set.height, data_set.width))
a = np.maximum(a, 0)
a = block_reduce(a, block_size=(blockSize, blockSize), func=np.sum)

nClusters = int(round(np.sum(a) / size))
print(nClusters)

b = np.where(a > 0)
print(a.size, b[0].size)

points, weights = np.dstack(b)[0], a[b].flatten()
print(len(points), len(weights))

cluster = KMeans(n_clusters=nClusters).fit(points, sample_weight=weights)
print(cluster.n_iter_)

centroids = []
for i, j in cluster.cluster_centers_:
    centroids.append([left + j * step, bottom + (h - i) * step])

result = {
  "type": "FeatureCollection",
  "features": [{
      "type": "Feature",
      "geometry": {
        "type": "MultiPoint",
        "coordinates": centroids
      },
      "properties": {}
    }]
}

json.dump(result, open('./data/output/md-' + '-'.join([str(year), str(size), str(blockSize)]) + '.json', 'w'), indent=2)


print(time.time() - start)