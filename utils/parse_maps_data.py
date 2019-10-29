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
        return float(self.line[4]) < float(other.line[4])


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
        file_streams[file] = csv.reader(open(str(input_path + file)), delimiter=' ')

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
            my_df.to_csv(outfile, index=False, header=False, sep=" ")
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
    my_df.to_csv(outfile, index=False, header=False, sep=" ")


def initial_split(writepath, readpath, verbose=True):
    """
    Reads the gzip and splits the data into multiple sorted buckets
    :param writepath: Directory to write the data
    :param readpath: gZip File
    :param verbose: Verbose boolean
    """
    print("\n\nReading Maps CSV and splitting into multiple sorted buckets\n\n")
    j = 1
    data = []
    fileNumber = 0

    with open(readpath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for current_line in readCSV:
            try:
                float(current_line[4])
                data.append(current_line)
                j += 1
            except ValueError:
                print("Not a float: " + current_line[4])
            if j % 2000000 == 0 and j != 0:
                if verbose:
                    print("Processing data: " + str(j))
                    data.sort(key=lambda data: float(data[4]))
                my_df = pd.DataFrame(data)
                my_df.to_csv('{}/data_{}.csv'.format(writepath, fileNumber), index=False, header=False,
                             sep=" ")
                data.clear()
                fileNumber += 1
    if verbose:
        print("Total rows: " + str(j))
    data.sort(key=lambda data: float(data[4]))
    my_df = pd.DataFrame(data)
    my_df.to_csv('{}/data_{}.csv'.format(writepath, fileNumber), index=False, header=False, sep=" ")


def get_keys(input_path, writepath, verbose = True):
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
    last = -181
    for file in files:
        with open(input_path + file[0]) as csvfile:
            if verbose:
                print("Working on file: ", file[0])
            readCSV = csv.reader(csvfile, delimiter=' ')
            for row in readCSV:
                if float(row[4]) > last:
                    writer.writerow([val, float(row[4])])
                    val += 1
                    last = float(row[4])
                elif float(row[4]) == last:
                    val += 1
                else:
                    print("Error: ", last, float(row[4]))


initial_split(writepath="/media/yash/Data/CSE662_Data/Maps/data_1", readpath="/media/yash/Data/CSE662_Data/Maps/Maps_Data.csv", verbose=True)
sort_merge(input_path="/media/yash/Data/CSE662_Data/Maps/data_1/", output_path='/media/yash/Data/CSE662_Data/Maps/data_2/', out_file_len=2000000)
get_keys(input_path='/media/yash/Data/CSE662_Data/Maps/data_2/', writepath="/media/yash/Data/CSE662_Data/Maps/")
