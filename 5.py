# reading geotiff

import numpy as np
import json
import rasterio
import time
from sklearn.cluster import MiniBatchKMeans
from skimage.measure import block_reduce


def get_centroids(year, population, size):
    block_size = 3

    image_path = './data/img/mda_ppp_' + str(year) + '.tif'
    data_set = rasterio.open(image_path)

    left, bottom, right, top = data_set.bounds
    w, h = data_set.width // block_size, data_set.height // block_size
    step = (right - left) * block_size / data_set.width

    a = data_set.read().reshape((data_set.height, data_set.width))
    a = np.maximum(a, 0)
    a = block_reduce(a, block_size=(block_size, block_size), func=np.sum)

    n_clusters = int(round(population / size))

    b = np.where(a > 0)
    indices, weights = np.dstack(b)[0], a[b].flatten()
    print('total points', len(indices))

    cluster = MiniBatchKMeans(n_clusters=n_clusters, init_size=n_clusters * 100).fit(indices, sample_weight=weights)
    print('total iterations', cluster.n_iter_)

    centroids = []
    for i, j in cluster.cluster_centers_:
        centroids.append([left + j * step, bottom + (h - i) * step])
    return centroids


start = time.time()

census = [p * 1.16 * (10**3) for p in  [3384, 2805, 2681]]
years = [2004, 2014, 2019]

size = 10 ** 4
for i in range(3):
    year = years[i]
    population = census[i]
    points = get_centroids(year, population, size)
    print(year, size)
    features = [{
      "type": "Feature",
      "geometry": {
        "type": "MultiPoint",
        "coordinates": points
      },
      "properties": {
          "year": year,
          "detail": size,
          "total": len(points) * size
      }
    }]

    result = {
      "type": "FeatureCollection",
      "features": features
    }

    json.dump(result, open('./data/output/year-' + str(year) + '-size-' + str(int(size / 1000)) + 'k.json', 'w'), indent=2)


print(time.time() - start)