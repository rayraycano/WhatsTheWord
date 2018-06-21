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
from utils import run_experiment

LOGDIR = './logs'


def main(filename, run_id, logdir,
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
    run_dir = os.path.join(logdir, run_id, datetime.now().strftime('%Y%m%d-%H%M'))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(
            vocabulary_size=vocabulary_size, batch_size=batch_size,
            embedding_size=embedding_size, skip_window=skip_window,
            num_skips=num_skips, num_sampled=num_sampled, num_steps=num_steps,
            model_context=model_context, lookahead=lookahead, model_type=model_type, **kwargs
        ), f, indent=4, sort_keys=True)

    vocabulary = read_data(filename)
    print('Data size', len(vocabulary))

    # Build the dictionary and replace rare words with UNK token.
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)
    number_indices = len(data)
    train_count = math.floor(number_indices * 0.85)
    randomized_indices = list(range(len(data)))
    random.shuffle(randomized_indices)
    train_ids = randomized_indices[:train_count]
    val_ids = randomized_indices[train_count:]

    print("dictionary size: ", len(dictionary))
    print("reversed dictionary size should be same: ", len(reversed_dictionary))
    # Check if our words of interest are common words in rap literature
    words_to_check = ['nigga', 'friend', 'brother', 'brotha', 'homie', 'pal', 'dog', 'bitch', 'gangster', 'gangsta']
    for w in words_to_check:
        print('{} in dict: {}'.format(w, dictionary.get(w, 'N/A')))

    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])
    del vocabulary

    # Small test run here
    test_batch_size = 8
    # num_skips, skip_window, data, data_indices, model_context, rd, lookahead=True):
    batch, model_inputs, labels, model_labels = generate_batch(
        batch_size=test_batch_size,
        num_skips=2,
        skip_window=1,
        data=data,
        data_indices=list(range(8)),
        model_context=model_context,
        rd=reversed_dictionary,
        lookahead=lookahead,
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

    ######### TODO: Factor this out into a training method for the conv net
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[None])
            train_model_inputs = tf.placeholder(tf.int32, shape=[None, model_context * (lookahead + 1)])
            train_labels = tf.placeholder(tf.int32, shape=[None, 1])
            train_model_labels = tf.placeholder(tf.int32, shape=[None])
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
            one_hot_labels = tf.one_hot(train_model_labels, vocabulary_size)
            print(one_hot_labels)
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=train_model_labels))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     logits=out,
            #     labels=one_hot_labels))
            idxs = tf.argmax(out, axis=1)
            print("idxs shape", idxs.shape)
            print(train_model_labels.shape)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(idxs, tf.cast(train_model_labels, tf.int64)), tf.float32))
            print('accuracy tensor: {}'.format(accuracy))

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
        val_inputs, val_model_inputs, val_labels, val_model_labels = generate_batch(
            batch_size=len(val_ids),
            skip_window=skip_window,
            num_skips=num_skips,
            data=data,
            data_indices=val_ids,
            model_context=model_context,
            rd=reversed_dictionary,
            lookahead=lookahead,
        )
        val_feed_dict = {
            train_inputs: val_inputs,
            train_labels: val_labels,
            train_model_labels: val_model_labels,
            train_model_inputs: val_model_inputs,
        }
        for step in range(1, num_steps+1):
            batch_inputs, model_inputs, batch_labels, model_labels = generate_batch(
                batch_size=batch_size,
                skip_window=skip_window,
                num_skips=num_skips,
                data=data,
                data_indices=train_ids,
                model_context=model_context,
                rd=reversed_dictionary,
                lookahead=lookahead,
            )

            feed_dict = {
                train_inputs: batch_inputs,
                train_labels: batch_labels,
                train_model_labels: model_labels,
                train_model_inputs: model_inputs,
            }
            run_metadata = tf.RunMetadata()
            _, _, summary, loss_result, cost_result, acc_result, preds, full_pred, flat_res = session.run(
                [nce_optimizer, cost_optimizer, merged, nce_loss, cost, accuracy, idxs, out, model.flattened],
                feed_dict=feed_dict,
                run_metadata=run_metadata
            )

            average_loss += loss_result
            average_cost += cost_result
            writer.add_summary(summary, step)
            # if step % 100 == 0:
                # print('Argmax Flattened\n', '\n'.join([str(np.argmax(x)) for x in flat_res[:10]]))
                # print('Full Preds\n', '\n'.join([str(x[:10]) for x in full_pred[:10]]))
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    average_cost /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                print('Average cost at step ', step, ': -----------', average_cost)
                print('Accuracy at step ', step, ': ', acc_result * 100, '%')
                print(batch_labels[0])
                print(batch_labels.shape)
                print(preds.shape)
                for i in range(len(preds)):
                    if i <= 10:
                        print("expected {}:{} predicted {}:{}".format(
                            model_labels[i],
                            reversed_dictionary[model_labels[i]],
                            preds[i],
                            reversed_dictionary[preds[i]]),
                        )
                    break
                average_loss = 0
                average_cost = 0
            if step % 1000 == 0:
                validation_accuracy = session.run(
                    accuracy,
                    feed_dict=val_feed_dict,
                )
                print("Validation Accuracy at step{}: {}%".format(step, validation_accuracy * 100))
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

    plot_tsne(final_embeddings, min(500, vocabulary_size), reversed_dictionary, run_dir)


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


def generate_batch(batch_size, num_skips, skip_window, data, data_indices, model_context, rd, lookahead=True):
    """
    Again, scooped up from the tensorflow tutorial
    :param batch_size: Number of examples to generate
    :param num_skips: How many times to reuse an input to generate a label.
    :param skip_window: How many words to consider left and right.
    :param data: dataset
    :param data_index: where to satart generating the batch
    :param lookahead: whether or not to use words following the input word for context
    :param rd: reveresed_dictionary for debugging purposes
    :return:
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    examples = np.random.choice(data_indices, batch_size // num_skips)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    model_batch = np.ndarray(shape=(batch_size, model_context * (1 + lookahead)), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    model_labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    model_span = 2 * model_context + 1

    for i, data_index in enumerate(examples):
        window_buffer = collections.deque(maxlen=span)
        model_buffer = collections.deque(maxlen=model_span)
        if data_index + span > len(data):
            data_index = 0
        window_buffer.extend(data[data_index:data_index + span])
        model_buffer.extend(data[data_index:data_index + model_span])
        data_index += span
        # for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
          batch[i * num_skips + j] = window_buffer[skip_window]
          labels[i * num_skips + j, 0] = window_buffer[context_word]
          model_batch[i*num_skips+j] = _make_model_example(list(model_buffer), model_span, lookahead)
          model_labels[i * num_skips + j] = model_buffer[model_span // 2]
          # print("target word: ", rd[model_labels[i*num_skips + j]])
          # print("context: ", [rd[x] for x in model_batch[i*num_skips+j]])
        if data_index == len(data):
          window_buffer.extend(data[0:span])
          # data_index = span
          data_index = np.random.randint(model_context, len(data) - model_context)
          window_buffer.extend(data[data_index:data_index + span])
          model_buffer.extend(data[data_index:data_index + model_span])
        else:
          # window_buffer.append(data[data_index])
          # model_buffer.append(data[data_index])
          # data_index += 1
          data_index = np.random.randint(model_context, len(data) - model_context)
          window_buffer.extend(data[data_index:data_index + span])
          model_buffer.extend(data[data_index:data_index + model_span])
    # Backtrack a little bit to avoid skipping words in the end of a batch
    model_batch = np.array(model_batch)
    # print("model batch: ", model_batch)
    return batch, model_batch, labels, model_labels


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--pfile")  # parameters file

    args = parser.parse_args()
    run_experiment(args, main)
