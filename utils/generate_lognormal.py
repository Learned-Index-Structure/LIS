from operator import itemgetter
from queue import PriorityQueue
import os
import pandas as pd
import csv
import numpy as np

class Data:
    """
        Class Data used for On Disk Sorting. The comparator is defined here
        :param line: Row of data
        :param filename: Name of the file to which row belongs
        """
    def __init__(self, line, filename):
        self.line = line
        self.filename = filename

    def __lt__(self, other):
        """
        Class Data used for On Disk Sorting. The comparator is defined here
        :param other: Compares other and self object
        :return: bool: Comparison at index 2.
        """
        return int(self.line[0]) < int(other.line[0])


def sort_merge(input_path, output_path):
    """
    Uses class Data's comparator and merges multiple files in the directory using Priority Queue
    :param input_path: Row of data
    :param output_path: Name of the file to which row belongs
    """
    print("Merging all the files")

    q = PriorityQueue()
    file_streams = dict()

    for file in os.listdir(input_path):
        file_streams[file] = csv.reader(open(str(input_path + file)), delimiter=' ')

    for k, v in file_streams.items():
        q.put(Data(next(v), k))

    f = open(output_path + "out_0.csv", 'w')

    i = 0
    while not q.empty():
        data = q.get()

        f.write(str(i) + " " + str(data.line[0]) + '\n')
        data_file = data.filename
        i += 1

        if i % 10000000 == 0:
            print("Generated number of rows: " + str(i))
        try:
            line = next(file_streams[data_file])
        except StopIteration:
            line = None
            file_streams.pop(data_file)

        if line:
            try:
                q.put(Data(line, data_file))
            except():
                pass


def initial_split(writepath):
    """
    Uses np random to generate the data into multiple sorted buckets
    :param writepath: Directory to write the data
    """
    print("\n\nGenerating 100,000,000 rows and saving in multiple files for merging later\n\n")

    np.random.seed(0)
    scale = 1000000000
    max = 2147483647 / scale
    nElements = 200000000

    i = 0
    while i < (nElements/100000000)*2:
        data = set()
        while len(data) < 100000000:
            s = np.random.lognormal(0, 2)
            if s > max:
                continue
            data.add(int(s * scale))
            if len(data) % 10000000 == 0:
                print("Generated samples: " + str(len(data)))
        data = sorted(data)
        f = open('{}/data_{}.csv'.format(writepath, i), 'w')

        print("\nData sorted. Writing to file.\n")

        for x in data:
            f.write(str(x) + '\n')
        f.close()
        data.clear()
        i += 1


def delete_duplicates(input_path, writepath, verbose = True):
    """
    Reads all the sorted data and writes just the key along with the row number to separate file
    :param input_path: The directory of sorted data
    :param writepath: Directory where the training data for model will be written
    :param verbose: Verbose boolean
    """
    print("\n\nReading all files and writing the keys in sorted manner.\n\n")

    files = []
    val = 0

    writer = csv.writer(open(writepath + "sorted_keys_non_repeated.csv", "w"), delimiter=' ')

    for file in os.listdir(input_path):
        files.append([file, int((file.split("_")[1]).split(".")[0])])
    files = sorted(files, key=itemgetter(1))
    last = -1
    for file in files:
        with open(input_path + file[0]) as csvfile:
            if verbose:
                print("Working on file: ", file[0])
            readCSV = csv.reader(csvfile, delimiter=' ')
            for row in readCSV:
                if int(row[1]) > last:
                    writer.writerow([val, int(row[1])])
                    val += 1
                    if val == 190000000:
                        break
                    last = int(row[1])
                elif int(row[1]) < last:
                    print("Error: ", last, int(row[1]))

initial_split(writepath="/media/yash/Data/CSE662_Data/LogNormal/data_1")
sort_merge(input_path="/media/yash/Data/CSE662_Data/LogNormal/data_1/", output_path='/media/yash/Data/CSE662_Data/LogNormal/data_2/')
delete_duplicates(input_path='/media/yash/Data/CSE662_Data/LogNormal/data_2/', writepath="/media/yash/Data/CSE662_Data/LogNormal/")

