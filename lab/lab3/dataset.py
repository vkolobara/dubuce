import numpy as np


class Dataset(object):

    def __init__(self, batch_size, sequence_length):
        self.char2id = dict()
        self.id2char = dict()
        self.x = None
        self.sorted_chars = None
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_batches = None

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()  # python 2

            # count and sort most frequent characters

            self.sorted_chars = list()

            for letter in set(data):
                self.sorted_chars.append((data.count(letter), letter))

            self.sorted_chars = sorted(self.sorted_chars, key=lambda x: x[0], reverse=True)

            self.sorted_chars = [x[1] for x in self.sorted_chars]

            # self.sorted chars contains just the characters ordered descending by frequency
            self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
            # reverse the mapping
            self.id2char = {k: v for v, k in self.char2id.items()}
            # convert the data to ids
            self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return [self.char2id[c] for c in sequence]

    def decode(self, sequence):
        # returns the sequence decoded as letters
        return [self.id2char[i] for i in sequence]

    def create_minibatches(self):
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))

        



dataset = Dataset()
dataset.preprocess("test.txt")

print(dataset.sorted_chars)