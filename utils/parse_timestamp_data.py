from pathlib import Path
import gzip
import csv
from operator import itemgetter
import pandas as pd

j = 0
data = []
fileNumber = 0
for filename in Path("/home/yash/Desktop/CSE-662/Data/2014").glob('**/*.gz'):
    with gzip.open(filename, 'r') as fin:
        i = 0
        for line in fin:
            if i != 0:
                data.append(line.decode().split('\t'))
            else:
                i += 1
            if j % 1000000 == 0 and j != 0:
                print(j)
                sorted(data, key=itemgetter(2))
                my_df = pd.DataFrame(data)
                my_df.to_csv('/media/yash/Data/aaa/data_{}.csv'.format(fileNumber), index=False, header=False, sep="|")
                data.clear()
                fileNumber += 1
            j += 1

print(j)
sorted(data, key=itemgetter(2))
my_df = pd.DataFrame(data)
my_df.to_csv('/media/yash/Data/aaa/data_{}.csv'.format(fileNumber), index=False, header=False, sep="|")