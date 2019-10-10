from queue import PriorityQueue

pos = 2
q = PriorityQueue()
file_streams = dict()

path = '../data/weblog/'
files = path + 'files.txt'  # List of file name to be merged


class Data:
    def __init__(self, line, filename):
        self.line = line
        self.filename = filename

    def __lt__(self, other):
        a = self.line.split("\t")
        b = other.line.split("\t")
        return float(a[pos]) < float(b[pos])


def get_file_stream():
    with open(files) as file:
        all_files = file.read().splitlines()
        for line in all_files:
            f = open(str(path + line), "r", encoding="utf-8")
            file_streams[line] = f


def init_pq():
    for k, v in file_streams.items():
        q.put(Data(v.readline(), k))


def sort_merge():
    out_file_len = 4
    i = 0
    file_suffix = 0
    outfile = path + "out" + str(file_suffix) + ".csv"
    f = open(outfile, "a+")

    while not q.empty():
        data = q.get()

        if i < out_file_len:
            f.write(data.line)
            i += 1
        else:
            f.close()
            file_suffix += 1
            outfile = path + "out" + str(file_suffix) + ".csv"
            f = open(outfile, "a+")
            f.write(data.line)
            i = 1

        data_file = data.filename
        line = file_streams[data_file].readline()
        if line:
            q.put(Data(line, data_file))

    if i != 0:
        f.close()


get_file_stream()
init_pq()
sort_merge()
