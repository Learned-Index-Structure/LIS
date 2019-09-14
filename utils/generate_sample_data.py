#!/usr/bin/python3

import sys
import random

if (len(sys.argv) != 3):
    print("./generate_sample_data.py [filename] [datasize in MB]")
    sys.exit()

fileName = str(sys.argv[1])
dataSize = str(sys.argv[2])

itemCount = (int(dataSize) * 1024 * 128)

print("Writing " + str(itemCount) + " entries to " + fileName)

f = open(fileName,"w+")

for i in range(itemCount):
    f.write(str(i) + " " + str(random.randint(i*10, (i+1)*10)) + "\n")