from pathlib import Path
import gzip
from operator import itemgetter
from queue import PriorityQueue
import os
import pandas as pd
import csv


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
        return float(self.line[2]) < float(other.line[2])


def sort_merge(input_path, output_path, out_file_len=1000000, verbose = True):
    """
    Uses class Data's comparator and merges multiple files in the directory using Priority Queue
    :param input_path: Row of data
    :param output_path: Name of the file to which row belongs
    :param out_file_len: Maximum number of rows in each file
    :param verbose: Verbose boolean
    """
    q = PriorityQueue()
    file_streams = dict()

    for file in os.listdir(input_path):
        file_streams[file] = csv.reader(open(str(input_path + file)), delimiter='|')

    for k, v in file_streams.items():
        q.put(Data(next(v), k))

    i = 0
    file_suffix = 0
    outfile = output_path + "out_" + str(file_suffix) + ".csv"
    f = open(outfile, "a+")

    data_block = []
    while not q.empty():
        data = q.get()

        if i < out_file_len:
            data_block.append(data.line)
            i += 1
        else:
            my_df = pd.DataFrame(data_block)
            my_df.to_csv(outfile, index=False, header=False, sep="|")
            file_suffix += 1
            if verbose:
                print("Writing file: " + outfile)
            outfile = output_path + "out_" + str(file_suffix) + ".csv"
            data_block.clear()
            data_block.append(data.line)
            i = 1

        data_file = data.filename
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

    if verbose:
        print("Writing file: " + outfile)
    my_df = pd.DataFrame(data_block)
    my_df.to_csv(outfile, index=False, header=False, sep="|")


def initial_split(writepath, readpath, verbose=True):
    """
    Reads the gzip and splits the data into multiple sorted buckets
    :param writepath: Directory to write the data
    :param readpath: gZip File
    :param verbose: Verbose boolean
    """
    print("\n\nReading from gzip and creating sorted files of 1M rows each.\n\n")
    j = 1
    data = []
    fileNumber = 0
    for filename in Path(readpath).glob('**/*.gz'):
        with gzip.open(filename, 'rt') as fin:
            for line in fin:
                if line[-1] == "\n":
                    line = line[:-1]
                current_line = line.split('\t')
                if len(current_line) == 9:
                    data.append(current_line)

                if j % 1000000 == 0 and j != 0:
                    if verbose:
                        print("Processing data: " + str(j))
                    data = sorted(data, key=itemgetter(2))
                    my_df = pd.DataFrame(data)
                    my_df.to_csv('{}/data_{}.csv'.format(writepath, fileNumber), index=False, header=False,
                                 sep="|")
                    data.clear()
                    fileNumber += 1
                j += 1
    if verbose:
        print("Total rows: " + str(j))
    data = sorted(data, key=itemgetter(2))
    my_df = pd.DataFrame(data)
    my_df.to_csv('{}/data_{}.csv'.format(writepath, fileNumber), index=False, header=False, sep="|")


def get_keys(input_path, writepath, verbose = True):
    """
    Reads all the sorted data and writes just the key along with the row number to separate file
    :param input_path: The directory of sorted data
    :param writepath: Directory where the training data for model will be written
    :param verbose: Verbose boolean
    """
    print("\n\nReading all files and writing the keys in sorted manner.\n\n")

    keys = []
    files = []
    values = []
    val = 0
    for file in os.listdir(input_path):
        files.append([file, int((file.split("_")[1]).split(".")[0])])
    files = sorted(files, key=itemgetter(1))
    last = -1
    for file in files:
        with open(input_path + file[0]) as csvfile:
            if verbose:
                print("Working on file: ", file[0])
            readCSV = csv.reader(csvfile, delimiter='|')
            for row in readCSV:
                if float(row[2]) > last:
                    keys.append(float(row[2]))
                    values.append(val)
                    val += 1
                    last = float(row[2])
                elif float(row[2]) == last:
                    val += 1
                else:
                    print("Error: ", last, float(row[2]))
    # my_df = pd.DataFrame(values, keys)
    my_df = pd.DataFrame({'values': values,
     'keys': keys
    })

    my_df.to_csv(writepath + "sorted_keys_non_repeated.csv", index=False, header=False, sep="|")


# initial_split(writepath="/media/yash/Data/data_1", readpath="/home/yash/Desktop/CSE-662/Data/2014", verbose=True)
# sort_merge(input_path="/media/yash/Data/data_1/", output_path='/media/yash/Data/data_2/', out_file_len=1000000)
get_keys(input_path='/media/yash/Data/data_2/', writepath="/media/yash/Data/data_2/")
