### Reverse-MIPS Search Algorithm

Efificent Python Implementation of “Reverse Maximum Inner Product Search: How to efficiently find users who would like to buy my item?” of Recsys 2021

Python implementation which uses cython/openmp for efficient inferences

### Usage
```python

import numpy as np
import pysimpfer.simpfer

### Random dataset generation
n_users, n_items = 1000, 10000
dim = 15
Q = np.random.normal(0, 1.5, size=(n_users, dim)) # user_vectors
P = np.random.normal(0, 1.5, size=(n_items, dim)) # item_vectors

r = pysimpfer.simpfer.Simpfer(Q, P)

item_idx = 6
ret = r.rmips(item_idx, k=15)
np.dot(Q[np.array(list(ret), dtype=np.int32)], P[item_idx])
```



### Note: this implementation is not official
