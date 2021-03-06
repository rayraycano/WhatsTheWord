import abc
import tensorflow as tf


class Model:

    def __init__(self, input_tensor, **kwargs):
        self.input_tensor = input_tensor
        pass

    @abc.abstractclassmethod
    def load_tesnsors(self):
        return None

    @abc.abstractclassmethod
    def predict(self, X):
        """
        Given matrix X, predict the output class
        :param X: N x M x [S x ..]Input tensor or placeholder
        :return: y: N x output dimension
        """

    @abc.abstractclassmethod
    def train(self, X, y):
        """
        Trains a model to learn X, which corresponds to y
        :return: None
        """

    def save(self, saver, sess, name):
        """
        :return:
        """
        saver.save(sess, name)


    @staticmethod
    def load_saved(path):
        """
        Loads a saved model from the given path
        :param path: path to the model
        :return: model that behaves under the Model interface
        """
        pass


class LSTMModel(Model):

    def __init__(self, input_tensor, **kwargs):
        Model.__init__(self, input_tensor, **kwargs)
        self.embedding_size = kwargs['embedding_size']
        self.vocabulary_size = kwargs['vocabulary_size']
        self.n_hidden = kwargs['n_hidden']
        self.model_context = kwargs['model_context']
        self.batch_size = kwargs['batch_size']
        self.weights = self.biases = self.rnn_cell = self.outputs = self.out = None

    def load_tesnsors(self):
        """
        load computation graph and pass back the last
        :param input_placeholder:
        :param output_placeholder:
        :return:
        """
        with tf.name_scope('lstm_weights'):
            self.weights = [tf.Variable(tf.random_normal([self.n_hidden, self.vocabulary_size], stddev=0.01))]
        with tf.name_scope('lstm_biases'):
            self.biases = [tf.Variable(tf.random_normal([self.vocabulary_size]))]

        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        print('input tensor shape', self.input_tensor.shape)
        print('state size', self.rnn_cell.state_size)
        inputs = self.sequence_inputs()
        print(len(inputs))
        print(inputs[0].shape)

        self.outputs, states = tf.contrib.rnn.static_rnn(self.rnn_cell, inputs, dtype=tf.float32)

        self.out = tf.matmul(self.outputs[-1], self.weights[0]) + self.biases[0]

        return self.out

    def sequence_inputs(self):
        inputs = tf.split(self.input_tensor, self.model_context, 1)
        return [x[:, 0, :] for x in inputs]

    def get_trained_variables(self):
        return [
            x for x in tf.trainable_variables()
            if 'bias' not in x.name.lower() and
            'lstm' in x.name.lower()
        ]


class MultiLSTMModel(LSTMModel):

    def load_tensors(self):
        with tf.name_scope('lstm_weights'):
            self.weights = [tf.Variable(tf.random_normal([self.n_hidden, self.vocabulary_size], stddev=0.01))]
        with tf.name_scope('lstm_biases'):
            self.biases = [tf.Variable(tf.random_normal([self.vocabulary_size]))]
        self.rnn_cell = tf.contrib.rnn.MultiLSTMCell([
            tf.contrib.rnn.BasicLSTMCell(self.n_hidden),
            tf.contrib.rnn.BasicLSTMCell(self.n_hidden),
        ])
        inputs = self.sequence_inputs()

        self.outputs, states = tf.contrib.rnn.static_rnn(self.rnn_cell, inputs, dtype=tf.float32)

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
        self.context = kwargs['model_context'] * (1 + kwargs['lookahead'])
        self.window_size = kwargs['window_size']
        self.vocabulary_size = kwargs['vocabulary_size']
        self.step_size = kwargs['step_size']
        self.activation_length = self.context // self.step_size
        self.output_dim = kwargs['output_dim']
        self.weights = self.conv = self.biases = self.filter = \
            self.activations = self.out = self.flattened = None

    def load_tesnsors(self):

        with tf.name_scope('conv_weights'):
            self.weights = [
                tf.Variable(
                    tf.random_normal(
                        [self.window_size, self.embedding_size, self.output_dim],
                        stddev=0.01)
                ),
                tf.Variable(
                    tf.random_normal(
                        [self.output_dim * self.activation_length, self.vocabulary_size],
                        stddev=0.01)
                ),
            ]

            print(self.input_tensor)
            print(self.weights[0])
            print(self.weights  [1])
            # batch_size x (n_input / step_size) x output_dim
            self.conv = tf.nn.conv1d(self.input_tensor, self.weights[0], stride=self.step_size, padding='SAME')
            print(self.conv)

        with tf.name_scope('conv_biases'):
            self.biases = [
                tf.Variable(tf.random_normal([self.activation_length, self.output_dim])),
                tf.Variable(tf.random_normal([self.vocabulary_size])),
            ]
        with tf.name_scope('calulations'):
            print(self.biases[0])
            self.activations = tf.nn.relu(self.conv + self.biases[0])
            # self.pooled = tf.layers.max_pooling1d(
            #     self.activations,
            #     (self.step_size,),
            #     (self.step_size,),
            #     'SAME'
            # )
            # print("pooled tensor {}".format(self.pooled))
            flattened = tf.contrib.layers.flatten(self.activations)  # batch-size x (8 * output_dim)
            self.flattened = flattened
            print("flattened tensor", flattened)
            print(self.weights[1])
            print(self.biases[1])
            self.out = tf.matmul(flattened, self.weights[1]) + self.biases[1]
            print(self.out)
        return self.out

    def get_trained_variables(self):
        return [
            x for x in tf.trainable_variables()
            if 'bias' not in x.name.lower() and
            'conv' in x.name.lower()
        ]

    def predict(self, X):
        # Intialize Graph
        # Intialize Session
        # load tesnsors
        # Feed dict of X going into self.input tensor
        # return outputs with softmax classification
        pass

class SplitCNN(CNNModel):

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
        # Model Context will be symmetrically applied to the behind and forward convolutions
        self.context = kwargs['model_context']
        self.window_size = kwargs['window_size']
        self.vocabulary_size = kwargs['vocabulary_size']
        self.step_size = kwargs['step_size']
        self.activation_length = self.context // self.step_size
        self.output_channel_dim = kwargs['output_channel_dim']
        self.output_dim = kwargs['output_dim']
        self.weights = self.conv_pre = self.conv_post = self.biases = self.filter = \
            self.activations_pre = self.activations_post = self.out = self.synthesize = \
            self.flattened = None

    def load_tesnsors(self):
        with tf.name_scope('conv_weights'):
            self.weights = [
                tf.Variable(
                    tf.random_normal(
                        [self.window_size, self.embedding_size, self.output_channel_dim],
                        stddev=0.01)
                ),
                tf.Variable(
                    tf.random_normal(
                        [self.window_size, self.embedding_size, self.output_channel_dim],
                        stddev=0.01)
                ),
                tf.Variable(
                    tf.random_normal(
                        [self.output_channel_dim * 2 * self.activation_length, self.output_dim],
                        stddev=0.01)
                ),
                tf.Variable(tf.random_normal([self.output_dim, self.vocabulary_size], stddev=0.01)),
            ]

            print(self.input_tensor)
            for w in self.weights:
                print(w)
            pre_target, post_target = tf.split(self.input_tensor, num_or_size_splits=2, axis=1)
            print("pre_target shape: ", pre_target.shape)
            print("post_target shape: ", post_target.shape)
            # batch_size x (n_input / step_size) x output_dim
            self.conv_pre = tf.nn.conv1d(pre_target, self.weights[0], stride=self.step_size, padding='SAME')
            self.conv_post = tf.nn.conv1d(post_target, self.weights[1], stride=self.step_size, padding='SAME')

        with tf.name_scope('conv_biases'):
            self.biases = [
                tf.Variable(tf.random_normal([self.activation_length, self.output_channel_dim])),
                tf.Variable(tf.random_normal([self.activation_length, self.output_channel_dim])),
                tf.Variable(tf.random_normal([self.output_dim])),
                tf.Variable(tf.random_normal([self.vocabulary_size])),
            ]
        with tf.name_scope('calulations'):
            self.activations_pre = tf.nn.relu(self.conv_pre + self.biases[0])
            self.activations_post = tf.nn.relu(self.conv_post + self.biases[1])

            flattened_pre = tf.contrib.layers.flatten(self.activations_pre)  # batch-size x (8 * output_dim)
            flattened_post = tf.contrib.layers.flatten(self.activations_post)
            self.flattened = tf.concat((flattened_pre, flattened_post), axis=1)
            print(self.weights[1])
            print(self.biases[1])
            self.synthesize = tf.nn.relu(tf.matmul(self.flattened, self.weights[2]) + self.biases[2])
            self.out = tf.matmul(self.synthesize, self.weights[3]) + self.biases[3]

            print(self.out)
        return self.out