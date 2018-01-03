import numpy as np

class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(0, 1e-2, (hidden_size, sequence_length)) # ... input projection
        self.W = np.random.normal(0, 1e-2, (hidden_size, hidden_size)) # ... hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1))  # ... input bias

        self.V = np.random.normal(0, 1e-2, (vocab_size, hidden_size))   # ... output projection
        self.c = np.zeros((vocab_size, 1))  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(
            self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        cache = (h_prev, x)
        h_current = np.tanh(np.matmul(W, h_prev.T) + np.matmul(U.T, x.T) + b)

        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, cache


    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h, cache = None, None

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, cache