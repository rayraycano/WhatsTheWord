import abc
import tensorflow as tf


class Model:

    def __init__(self, input_tensor, **kwargs):
        self.input_tensor = input_tensor
        pass

    @abc.abstractclassmethod
    def load_tesnsors(self):
        return None


class LSTMModel(Model):

    def __init__(self, input_tensor, **kwargs):
        Model.__init__(self, input_tensor, **kwargs)
        self.embedding_size = kwargs['embedding_size']
        self.vocabulary_size = kwargs['vocabulary_size']
        self.n_hidden = kwargs['n_hidden']
        self.n_input = kwargs['n_input']
        self.state_size = (self.n_input, self.embedding_size)
        self.weights = self.biases = self.rnn_cell = self.outputs = self.out = None

    def load_tesnsors(self):
        """
        load computation graph and pass back the last
        :param input_placeholder:
        :param output_placeholder:
        :return:
        """
        with tf.name_scope('lstm_weights'):
            self.weights = [tf.Variable(tf.random_normal([self.n_hidden, self.vocabulary_size]))]
        with tf.name_scope('lstm_biases'):
            self.biases = [tf.Variable(tf.random_normal([self.vocabulary_size]))]

        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        self.outputs, states = tf.contrib.rnn.static_rnn(self.rnn_cell, self.input_tensor, dtype=tf.float32)

        self.out = tf.matmul(self.outputs[-1], self.weights[0]) + self.biases[0]

        return self.out

class CNNModel(Model):

    def __init__(self, input_tensor, **kwargs):
        """
        Initialize a Convolutional Nerual Net Model
        :param input_tensor: batch_size x window_size x embedding_size
        :param kwargs: dictionary specifying dimensions. The arguments separate form those already
        being passed in are
            window_size: filter size
            step_size: step size when sliding the filter
            output_dim: output channels after conv
            n_input: size of context window
        """
        Model.__init__(self, input_tensor, **kwargs)
        self.embedding_size = kwargs['embedding_size']
        self.n_input = kwargs['n_input'] # window size of words: start off with size 8
        self.window_size = kwargs['window_size']  # start off with 2
        self.vocabulary_size = kwargs['vocabulary_size']
        self.step_size = kwargs['step_size']
        self.output_dim = kwargs['output_dim']
        self.weights = self.conv = self.biases = self.filter = self.activations = self.out = None

    def load_tesnsors(self):

        with tf.name_scope('conv_weights'):
            self.weights = [
                tf.Variable(tf.random_normal([self.window_size, self.embedding_size, self.output_dim])),
                tf.Variable(tf.random_normal([self.output_dim * 8, self.vocabulary_size])),
            ]

            print(self.input_tensor)
            print(self.weights[0])
            # batch_size x 4 x output_dim
            self.conv = tf.nn.conv1d(self.input_tensor, self.weights[0], stride=self.step_size, padding='SAME')
            print(self.conv)

        with tf.name_scope('conv_biases'):
            self.biases = [
                tf.Variable(tf.random_normal([8, self.output_dim])),
                tf.Variable(tf.random_normal([self.vocabulary_size])),
            ]
        with tf.name_scope('calulations'):
            print(self.biases[0])
            self.activations = tf.nn.relu(self.conv + self.biases[0])
            flattened = tf.contrib.layers.flatten(self.activations) # batch-size x (8 * output_dim)
            print(flattened)
            print(self.weights[1])
            print(self.biases[1])
            self.out = tf.matmul(flattened, self.weights[1]) + self.biases[1]
            print(self.out)
        return self.out
