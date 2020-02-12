# geojson

import numpy as np
import json
import time


start = time.time()

a = np.random.random((12, 2)) * 85
print(a)

features = []
for coords in a:
    features.append({
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [float(coords[0]), float(coords[1])]
      },
      "properties": {}
    })

result = {
  "type": "FeatureCollection",
  "features": features
}

json.dump(result, open('./data/md.json', 'w'), indent=2)


print (time.time() - start)