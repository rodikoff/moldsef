# reading geotiff

import numpy as np
import rasterio
import time

start = time.time()

data = rasterio.open('./data/mda_ppp_2020.tif')
print(data.width, data.height)

a = data.read()
print(a[np.where(a > 0)])


print(time.time() - start)

