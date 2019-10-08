import numpy as np
import random as rand

np.random.seed(0) 

s = np.random.lognormal(0, 2, 1000) * 10000000

data = sorted(set(np.array(s, dtype=int)))

f = open('../data/log_normal.csv', 'w')

for x in data:
    f.write(str(x) + '\n')
