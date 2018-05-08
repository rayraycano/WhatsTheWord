import argparse
import tensorflow as tf
import collections
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
from datetime import datetime
import json
import models

from make_dataset import clean_data, get_regex_replacements

LOGDIR = './logs'


def main(filename, run_id,
         vocabulary_size=50000,
         batch_size=128,
         embedding_size=128,
         skip_window=1,
         num_skips=2,
         num_sampled=64,
         num_steps=50000,
         model_type='lstm',
         model_context=4,
         lookahead=True,
         **kwargs):
    """
    :param filename: Name of the data file
    :param run_id: Name of the run
    :param vocabulary_size: Size of recognized vocabulary words
    :param batch_size: number of examples in each batch
    :param embedding_size: Dimension of the embedding vector.
    :param skip_window: How many words to consider left and right.
    :param num_skips: How many times to reuse an input to generate a label.
    :param num_sampled: Number of negative examples to sample.
    :param num_steps: Number of training iterations to go through.
    :param model_type: type of model to run

    Saves the model, lookup metadata, and tsne clustering after training a model off
    of a word embedding.
    """
    run_dir = os.path.join(LOGDIR, run_id, datetime.now().strftime('%Y%m%d-%H%M'))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(
            vocabulary_size=vocabulary_size, batch_size=batch_size,
            embedding_size=embedding_size, skip_window=skip_window,
            num_skips=num_skips, num_sampled=num_sampled, num_steps=num_steps,
        ), f, indent=4, sort_keys=True)

    vocabulary = read_data(filename)
    print('Data size', len(vocabulary))

    # Build the dictionary and replace rare words with UNK token.
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)
    # Check if our words of interest are common words in rap literature
    words_to_check = ['nigga', 'friend', 'brother', 'brotha', 'homie', 'pal', 'dog', 'bitch', 'gangster', 'gangsta']
    for w in words_to_check:
        print('{} in dict: {}'.format(w, dictionary.get(w, 'N/A')))

    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])
    del vocabulary

    # Small test run here
    data_index = 0
    test_batch_size = 8
    batch, model_inputs, labels, data_index = generate_batch(
        test_batch_size, 2, 1, data, data_index, model_context, lookahead
    )
    print('batch: ', batch)
    print('labels: ', labels)
    for i in range(test_batch_size):
        print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0],
              reversed_dictionary[labels[i, 0]])

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_model_inputs = tf.placeholder(tf.int32, shape=[batch_size, 8])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # tensors = model.load_tensors()
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
                )
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                model_embed = tf.nn.embedding_lookup(embeddings, train_model_inputs)
            with tf.name_scope('model'):
                model = get_model(model_type, input_tensor=model_embed,
                                  vocabulary_size=vocabulary_size,
                                  embedding_size=embedding_size,
                                  skip_window=skip_window,
                                  num_skips=num_skips,
                                  num_sampled=num_sampled,
                                  num_steps=num_steps,
                                  model_context=model_context,
                                  lookahead=lookahead, **kwargs)
                out = model.load_tesnsors()
            with tf.name_scope('nce_weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        shape=[vocabulary_size, embedding_size],
                        stddev=1.0/math.sqrt(embedding_size),
                    )
                )

            with tf.name_scope('nce_biases'):
                nce_biases = tf.Variable(
                    tf.zeros([vocabulary_size])
                )

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))
            # Note: We use the train inputs here because that is the word we're trying to predict
            one_hot_labels = tf.one_hot(train_inputs, vocabulary_size)
            print(one_hot_labels)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=out,
                labels=one_hot_labels))

        tf.summary.scalar('nce_loss', nce_loss)
        with tf.name_scope('optimizer'):
            nce_optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)
            cost_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)

        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    # Train Now
    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(run_dir, session.graph)
        init.run()
        print('Started from the bottom')

        average_loss = 0
        average_cost = 0
        for step in range(num_steps):
            batch_inputs, model_inputs, batch_labels, new_data_index = generate_batch(
                batch_size=batch_size,
                skip_window=skip_window,
                num_skips=num_skips,
                data=data,
                data_index=data_index,
                model_context=model_context,
                lookahead=lookahead,
            )

            data_index = new_data_index
            feed_dict = {
                train_inputs: batch_inputs,
                train_labels: batch_labels,
                train_model_inputs: model_inputs,
            }
            run_metadata = tf.RunMetadata()
            _, _, summary, loss_result, cost_result = session.run(
                [nce_optimizer, cost_optimizer, merged, nce_loss, cost],
                feed_dict=feed_dict,
                run_metadata=run_metadata
            )
            average_loss += loss_result
            average_cost += cost_result
            writer.add_summary(summary, step)
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    average_cost /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                print('Average cost at step ', step, ': -----------', average_cost)
                average_loss = 0
                average_cost = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reversed_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reversed_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

        # Write corresponding labels for the embeddings.
        with open(os.path.join(run_dir, 'metadata.tsv'), 'w') as f:
            for i in range(vocabulary_size):
                f.write(reversed_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(run_dir, 'model.ckpt'))

        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(run_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    plot_tsne(final_embeddings, 500, reversed_dictionary, run_dir)


def plot_tsne(embeddings, plot_only, reversed_dictionary, run_dir):
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(run_dir, 'tsne.png'))


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)


def get_model(model_name, **kwargs):
    if model_name == 'lstm':
        return models.LSTMModel(**kwargs)
    if model_name == 'cnn':
        return models.CNNModel(**kwargs)
    raise Exception('Invalid Model Type: {}'.format(model_name))


def generate_batch(batch_size, num_skips, skip_window, data, data_index, model_context, lookahead=True):
    """
    Again, scooped up from the tensorflow tutorial
    :param batch_size: Number of examples to generate
    :param num_skips: How many times to reuse an input to generate a label.
    :param skip_window: How many words to consider left and right.
    :param data: dataset
    :param data_index: where to satart generating the batch
    :param lookahead: whether or not to use words following the input word for context
    :return:
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    model_batch = np.ndarray(shape=(batch_size, model_context * (1 + lookahead)), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    model_span = 2 * model_context + 1
    window_buffer = collections.deque(maxlen=span)
    model_buffer = collections.deque(maxlen=model_span)
    if data_index + span > len(data):
        data_index = 0
    window_buffer.extend(data[data_index:data_index + span])
    model_buffer.extend(data[data_index:data_index + model_span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
          batch[i * num_skips + j] = window_buffer[skip_window]
          labels[i * num_skips + j, 0] = window_buffer[context_word]
          model_batch[i*num_skips+j] = _make_model_example(list(model_buffer), model_span, lookahead)
        if data_index == len(data):
          window_buffer.extend(data[0:span])
          data_index = span
        else:
          window_buffer.append(data[data_index])
          data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    model_batch = np.array(model_batch)
    return batch, model_batch, labels, data_index


def _make_model_example(model_buffer, span, lookahead):
    if len(model_buffer) == span:
        first_half = model_buffer[0:span // 2]
        return first_half + model_buffer[(span // 2 + 1):] if lookahead else first_half
    raise Exception("Unexpected model_buffer Length {}".format(len(model_buffer)))
    # return window_buffer.copy()[1:] + [0] * (span - len(window_buffer))

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename, 'r') as f:
        data = f.read().split()
    return data


def build_dataset(words, n_words):
    """
    Process raw inputs into a dataset.
    Taken from "https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/word2vec"
    :returns
        data - list of codes (integers from 0 to vocabulary_size-1). This is the original text but words are
            replaced by their codes
        count - map of words(strings) to count of occurrences
        dictionary - map of words(strings) to their codes(integers)
        reverse_dictionary - maps codes(integers) to words(strings)
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
          unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_similarity(words, run_id, full_folderpath=None):
    run_config, dictionary, reversed_dictionary, full_folderpath = load_run_data(run_id, full_folderpath)
    if run_config is None:
        return None
    vocabulary_size = run_config['vocabulary_size']
    embedding_size = run_config['embedding_size']
    tf.reset_default_graph()
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    tensors = _load_tensors(vocabulary_size, embedding_size, graph)
    valid_words = [x + '\n' for x in words if x + '\n' in dictionary.keys()]
    print('valid words: {}'.format(valid_words))
    print('dictionary keys: {}'.format(list(dictionary.keys())[:10]))
    vectorized = [dictionary[x] for x in valid_words]
    with session as sess:
        predict_dataset = tf.constant(vectorized, dtype=tf.int32)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(full_folderpath, 'model.ckpt'))
        print('model resotred!')
        embeddings = tensors['embeddings']
        norm = tensors['norm']
        normalized_embeddings = tensors['normalized_embeddings']
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  predict_dataset)

        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        sim = similarity.eval()
        for i in range(len(vectorized)):
            valid_word = reversed_dictionary[vectorized[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reversed_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)



def _make_prediction_dataset(text, skip_window, word_map):
    """
    DEPRECATED for the time being due to a misunderstanding

    cleans text and converts it into a dataset
    :param text:
    :param skip_window:
    :param word_map:
    :return:
    """
    n_words = ['nigga', 'niggas', 'nigger', 'niggers']
    cleaned_text, err = clean_data(text, check_for_http=False)
    split_text = cleaned_text.split()
    if err:
        return []
    idxs = []
    vectorized = []
    # Find each index
    for idx, w in enumerate(split_text):
        if w in n_words:
            idxs.append(idx)
        maybe_num = word_map.get(w)
        vectorized.append(word_map['UNK'] if maybe_num is None else maybe_num)
    # TODO: Left of here: Test this function!

    # Create example at each index
    dataset = []
    for idx in idxs:
        instance_dataset = []
        print('generating example for index: {}'.format(idx))
        print('vecotrized: {}'.format(vectorized))
        context_idx = max(idx - skip_window, 0)
        while context_idx < idx:
            instance_dataset.append(vectorized[context_idx])
            context_idx += 1
        context_idx = min(idx + skip_window, len(vectorized) - 1)
        while context_idx > idx:
            instance_dataset.append(vectorized[context_idx])
            context_idx -= 1
        print('instance dataset for idx {}: {}'.format(idx, instance_dataset))
        dataset.append(instance_dataset)
    return dataset


def _predict(text, run_id, full_folderpath=None):
    """
    Predict what the target word should be from the cleaned text
    :param cleaned_text: text to predict on, with the expectation that replacement
        words are in there
    :param run_id: string that specifies the model to load
    :param full_folderpath: optional datetime stamp to specify the run in format YYYYMMDD-HHMM
    :return:
    """

    run_config, dictionary, reversed_dictionary, full_folderpath = load_run_data(run_id, full_folderpath)

    if not run_config is None:
        return None

    vocabulary_size = run_config['vocabulary_size']
    embedding_size = run_config['embedding_size']
    num_skips = run_config['num_skips']
    skip_window = run_config['skip_window']
    predict_data = [dictionary[w] for w in cleaned_text]
    predict_dataset = generate_batch(1, num_skips, skip_window, predict_data, 0)

    tf.reset_default_graph()
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    tensors = _load_tensors(vocabulary_size, embedding_size, graph)
    with session as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(full_folderpath, 'model.ckpt'))
        print('model resotred!')
        embeddings = tensors['embeddings']
        norm = tensors['norm']
        normalized_embeddings = tensors['normalized_embeddings']
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  predict_dataset)

        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

def load_run_data(run_id, full_folderpath=None):
    """
    Loads the run hyperparameters and dictionary mappings
    :param run_id: name of run
    :param full_folderpath: optional specification of full folderpath to pull run from
    :return:
        run_config: dictionary[string]:value -- specifies run hyperparameters
        dictionary: dictionary[string]: number -- maps words to lookup number
        reversed_dictionary: dictionary[number]: string -- maps numbers to vocab words
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_id_path = os.path.join(dir_path, 'logs', run_id)
    print(dir_path)
    print(run_id_path)
    if not os.path.isdir(run_id_path):
        print("invalid run_id: {}".format(run_id_path))
        return None, None, None, None
    if full_folderpath is None:
        datefolder_path = max(os.listdir(run_id_path))
        print("Using run: {}".format(datefolder_path))
        full_folderpath = os.path.join(run_id_path, datefolder_path)
    with open(os.path.join(full_folderpath, 'config.json'), 'r') as c:
        run_config = json.load(c)
    with open(os.path.join(full_folderpath, 'metadata.tsv'), 'r') as m:
        reversed_dictionary = {i: w for i, w in enumerate(m.readlines())}
    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))

    return run_config, dictionary, reversed_dictionary, full_folderpath

def _load_tensors(vocabulary_size, embedding_size, graph):
    """
    Loads the weights from the specified run, taking the most recent run to be the model
    :param full_folderpath: folderpath specifying where to load the model from
    :param vocabulary_size: size of voacbulary used for the model
    :param embedding_size: size of the embedding for the modle
    :param graph: tensorflow computation graph
    :return: mapping embedding, norm of embeddings, normalized embeddings
    """
    with graph.as_default():

        with tf.device('/cpu:0'):
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
                )
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        shape=[vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size),
                    )
                )
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(
                    tf.zeros([vocabulary_size])
                )
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
    return dict(
        embeddings=embeddings,
        nce_weights=nce_weights,
        nce_biases=nce_biases,
        norm=norm,
        normalized_embeddings=normalized_embeddings,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--pfile")  # parameters file

    args = parser.parse_args()
    param_file = args.pfile
    with open(param_file, 'r') as p:
        param_dict = json.load(p)
        datafile = param_dict.pop('datafile')
        run_id = param_file.split('/')[-1].split('.')[0]
        print('run_id: ' + run_id)
    main(datafile, run_id, **param_dict)
