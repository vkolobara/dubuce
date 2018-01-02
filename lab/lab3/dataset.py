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
        self.batches = None
        self.current_batch = -1

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
        batch_seq_size = self.batch_size*self.sequence_length
        self.num_batches = len(self.x) // batch_seq_size
        self.batches = list()

        x = self.x[: self.num_batches * batch_seq_size]

        if len(x) == self.num_batches * batch_seq_size:
            y = np.roll(x, -1)
        else:
            y = self.x[1: self.num_batches * batch_seq_size + 1]

        x = np.stack(np.split(x, self.batch_size))
        y = np.stack(np.split(y, self.batch_size))

        self.batches = np.array(list(zip(x, y)))

    def next_minibatch(self):
        self.current_batch = (self.current_batch + 1) % (self.num_batches*self.batch_size)
        batch_index = self.current_batch // self.num_batches
        seq_index = self.current_batch % self.num_batches

        new_epoch = self.current_batch == 0

        batch_x, batch_y = self.batches[batch_index]

        batch_x = batch_x[seq_index*self.sequence_length : (seq_index+1)*self.sequence_length]
        batch_y = batch_y[seq_index*self.sequence_length : (seq_index+1)*self.sequence_length]

        return new_epoch, batch_x, batch_y


dataset = Dataset(2, 3)
dataset.preprocess("test.txt")
dataset.create_minibatches()
print(dataset.sorted_chars)
print(dataset.next_minibatch())
print(dataset.next_minibatch())
print(dataset.next_minibatch())
print(dataset.next_minibatch())
print(dataset.next_minibatch())
print(dataset.next_minibatch())