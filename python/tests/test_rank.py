import alpha
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG)

a = np.random.rand(5000_0000)

alpha.set_ctx(groups=100)
t1 = time.time()
alpha.RANK(a)
t2 = time.time()
print(len(a), t2 - t1)
