import numpy as np


class RNN(object):
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(0, 1e-2, (vocab_size, hidden_size))  # ... input projection
        self.W = np.random.normal(0, 1e-2, (hidden_size, hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1))  # ... input bias

        self.V = np.random.normal(0, 1e-2, (hidden_size, vocab_size))  # ... output projection
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

        cache = dict()

        cache['x'] = x
        cache['h_prev'] = h_prev
        cache['W'] = W
        cache['U'] = U
        cache['b'] = b

        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b.T)
        cache['h_curr'] = h_current

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

        cache = dict()
        cache["ts"] = list()

        h = np.zeros((x.shape[0], self.sequence_length, self.hidden_size))

        h_prev = h0
        for t in range(self.sequence_length):
            x_t = x[:, t, :]
            h_prev, cache_t = self.rnn_step_forward(x_t, h_prev, U, W, b)
            cache['ts'].append(cache_t)
            h[:, t, :] = h_prev

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        dh_prev, dU, dW, db = None, None, None, None

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        dh_prev = (1 - cache['h_curr'] ** 2) * grad_next

        dU = np.dot(cache['x'].T, dh_prev)
        dW = np.dot(cache['h_prev'].T, dh_prev)
        db = np.sum(dh_prev, axis=0)[:, None]

        dh_prev = np.dot(dh_prev, self.W.T)

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU, dW, db = None, None, None

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        dh_prev = np.zeros_like(dh[:, 0, :])

        for t in reversed(range(self.sequence_length)):
            grad_next = dh[:, t, :] + dh_prev
            cache_t = cache['ts'][t]

            dh_prev, dU_t, dW_t, db_t = self.rnn_step_backward(grad_next, cache_t)

            dU += dU_t
            dW += dW_t
            db += db_t

        for param in [dU, dW, db]:
            np.clip(param, -5, 5, out=param)

        return dU, dW, db

    def output(self, h, V, c):
        # Calculate the output probabilities of the network
        o = np.empty((h.shape[0], self.sequence_length, self.vocab_size))

        for t in range(h.shape[1]):
            o[:, t, :] = np.dot(h[:, t, :], V) + c.T

        return o

    def output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        loss, dh, dV, dc = None, None, None, None

        # calculate the output (o) - unnormalized log probabilities of classes
        o = self.output(h, V, c)
        # calculate yhat - softmax of the output

        yhat = np.empty_like(o)

        loss = 0

        for i in range(o.shape[1]):
            x = np.exp(o[:, i, :])
            exp_shift = np.exp(x - np.max(x, axis=1, keepdims=True))
            yhat[:, i, :] = x / np.sum(x, axis=1, keepdims=True)
            yhat[:, i, :] = np.clip(yhat[:, i, :], 1e-15, 1 - 1e-15)

            loss += np.mean(np.log(np.sum(x, axis=1)) - np.sum(y[:, i, :] * o[:,i,:], axis=1))

        # loss = -np.mean(y * np.log(yhat))
        # loss = -np.sum(y*np.log(yhat))

        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        dy = yhat - y

        # calculate the gradients with respect to the output parameters V and c
        dV = np.einsum('nth,ntv->hv', h, dy)
        # dV = np.dot(dy.T, h)

        # dc = np.sum(dy, axis=(0,1,2))
        dc = np.einsum('ntv->v', dy)[:, None]

        for param in [dV, dc]:
            np.clip(param, -5, 5, out=param)

        # calculate the gradients with respect to the hidden layer h
        dh = np.empty_like(h)

        last_dh = np.zeros_like(h[:, 0, :])

        for i in reversed(range(self.sequence_length)):
            dh[:, i, :] = np.dot(dy[:, i, :], V.T)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc):
        eta = 1e-8
        # update memory matrices
        self.memory_U += dU*dU
        self.memory_W += dW*dW
        self.memory_V += dV*dV
        self.memory_b += db*db
        self.memory_c += dc*dc

        # perform the Adagrad update of parameters
        self.U -= dU * self.learning_rate / np.sqrt(self.memory_U + eta)
        self.W -= dW * self.learning_rate / np.sqrt(self.memory_W + eta)
        self.V -= dV * self.learning_rate / np.sqrt(self.memory_V + eta)
        self.b -= db * self.learning_rate / np.sqrt(self.memory_b + eta)
        self.c -= dc * self.learning_rate / np.sqrt(self.memory_c + eta)

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc)

        return loss, h[:, -1, :]

    def sample(self, seed, n_sample):
        h0, seed_onehot, sample = None, None, None
        # inicijalizirati h0 na vektor nula
        # seed string pretvoriti u one-hot reprezentaciju ulaza

        h0 = np.zeros((1, self.hidden_size))
        seed_onehot = np.eye(self.vocab_size)[seed][:]

        h_prev = self.h0[-1][None, :]

        x = list()
        #h_prev = self.h0
        print(seed_onehot.shape)
        for t in range(seed_onehot.shape[0]):
            h_prev = np.tanh(np.dot(seed_onehot[t], self.U) + np.dot(h_prev, self.W) + self.b.T)

        out = self.output(h_prev[:, None, :], self.V, self.c)[-1, -1, :][None, :]
        index = np.argmax(out)

        out = np.zeros((1, self.vocab_size))
        out[:, index] = 1

        for t in range(n_sample - len(seed)):

            h_t = np.tanh(np.dot(out, self.U) + np.dot(h_prev, self.W) + self.b.T)
            out = np.dot(h_t, self.V) + self.c.T

            x.append(np.argmax(out))
            out = np.zeros((1, self.vocab_size))

            out[:, x[-1]] = 1
            #out = np.argmax(out, axis=1)

            h_prev = h_t

        return x


from random import uniform


def lossFun(inputs, targets, h0, rnn):
    h, cache = rnn.rnn_forward(inputs, h0, rnn.U, rnn.W, rnn.b)
    loss, dh, dWhy, dby = rnn.output_loss_and_grads(h, rnn.V, rnn.c, targets)
    dWxh, dWhh, dbh = rnn.rnn_backward(dh, cache)

    return loss, dWhy, dby, dWxh, dWhh, dbh, h


def gradCheck(inputs, targets, h0):
    rnn = RNN(100, 3, 4, 0.01)

    num_checks, delta = 20, 1e-7

    _, dWhy, dby, dWxh, dWhh, dbh, _ = lossFun(inputs, targets, h0, rnn)

    Wxh = rnn.U
    Whh = rnn.W
    Why = rnn.V
    bh = rnn.b
    by = rnn.c

    for param, dparam, name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby],
                                   ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)
        print(name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _ = lossFun(inputs, targets, h0, rnn)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _ = lossFun(inputs, targets, h0, rnn)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            # rel_error should be on order of 1e-7 or less


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(hidden_size=hidden_size, sequence_length=sequence_length, vocab_size=vocab_size,
              learning_rate=learning_rate)

    current_epoch = 0
    batch = 0
    batch_size = dataset.batch_size

    h0 = np.zeros((batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = None, None

        x_oh = np.empty((x.shape[0], x.shape[1], vocab_size))
        y_oh = np.empty((y.shape[0], y.shape[1], vocab_size))

        for i in range(x.shape[0]):
            x_oh[i] = np.eye(vocab_size)[x[i]]

        for i in range(y.shape[0]):
            y_oh[i] = np.eye(vocab_size)[y[i]]

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)
        if e:
            print("Loss at epoch %d: %f" % (current_epoch, loss))

        if batch % 100 == 0:
            print("Loss at batch %d: %f" % (batch, loss))

        if batch % sample_every == 0:
            # run sampling (2.2)
            seed = 'HAN:\nIs that good or bad?\n\n'
            print("==============================================")
            print(seed)
            print("SAMPLE:")
            rnn.h0 = h0
            ids = rnn.sample(dataset.encode(seed), 300)
            print(''.join(dataset.decode(ids)))
            print("==============================================")

        batch += 1
