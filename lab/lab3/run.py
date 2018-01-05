import rnn as rnn
import dataset as dataset

hidden_size = 100
vocab_size = 71
sequence_length = 25
learning_rate = 1e-1
batch_size = 1
input_file = "data/selected_conversations.txt"

ds = dataset.Dataset(batch_size=batch_size,
                     sequence_length=sequence_length)
ds.preprocess(input_file=input_file)
ds.create_minibatches()

rnn.run_language_model(ds, max_epochs=10,
                       hidden_size=hidden_size,
                       sequence_length=sequence_length,
                       learning_rate=learning_rate)
