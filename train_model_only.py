import argparse
import tensorflow as tf
import numpy as np
import random
import math
import os
from tensorflow.contrib.tensorboard.plugins import projector
from datetime import datetime
import json
import models
from utils import run_experiment, run_grid_search
from embedding_only import get_tensors, load_run_data
from word2vec import generate_batch

LOGDIR = './logs'
NUM_SKIPS = 1

def main(filename, run_id, logdir,
         embedding_dir=None,
         batch_size=128,
         skip_window=1,
         num_steps=50000,
         model_type='lstm',
         model_context=4,
         lookahead=True,
         l2_lambda=0.1,
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
    if not embedding_dir:
        raise Exception("Need directory of embedding")
    run_dir = os.path.join(logdir, run_id, datetime.now().strftime('%Y%m%d-%H%M'))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    embedding_params, dictionary, reversed_dictionary, full_folderpath = \
        load_run_data(embedding_dir, embedding_dir)
    vocabulary_size = embedding_params['vocabulary_size']
    embedding_size = embedding_params['embedding_size']
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(
            vocabulary_size=vocabulary_size, batch_size=batch_size,
            embedding_size=embedding_size, skip_window=skip_window,
            num_skips=NUM_SKIPS, num_steps=num_steps,
            model_context=model_context, lookahead=lookahead, model_type=model_type, l2_lambda=l2_lambda,
            embedding_dir=embedding_dir,
            **kwargs,
        ), f, indent=4, sort_keys=True)

    vocabulary = read_data(filename)
    print('Data size', len(vocabulary))

    data = tokenize(vocabulary, dictionary)

    number_indices = len(data)
    train_count = math.floor(number_indices * 0.85)

    # Sample from our data to create batches
    randomized_indices = list(range(len(data)))
    random.shuffle(randomized_indices)

    # Split into Train and Validation
    train_ids = randomized_indices[:train_count]
    val_ids = randomized_indices[train_count:]

    print("dictionary size: ", len(dictionary))
    print("reversed dictionary size should be same: ", len(reversed_dictionary))
    del vocabulary

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # TODO: Factor this out into a training method for the conv net
    tf.reset_default_graph()
    graph = tf.Graph()
    tensors = get_tensors(vocabulary_size, embedding_size, graph, full_folderpath)
    embeddings = tensors['embeddings']

    with graph.as_default():
        with tf.name_scope('inputs'):
            train_model_inputs = tf.placeholder(tf.int32, shape=[None, model_context * (lookahead + 1)])
            train_model_labels = tf.placeholder(tf.int32, shape=[None])

        with tf.device('/cpu:0'):
            model_embed = tf.nn.embedding_lookup(embeddings, train_model_inputs)
            with tf.name_scope('model'):
                model = get_model(model_type, input_tensor=model_embed,
                                  vocabulary_size=vocabulary_size,
                                  embedding_size=embedding_size,
                                  skip_window=skip_window,
                                  num_skips=NUM_SKIPS,
                                  num_steps=num_steps,
                                  model_context=model_context,
                                  lookahead=lookahead,
                                  batch_size=batch_size, **kwargs)
                out = model.load_tesnsors()

        with tf.name_scope('loss'):
            # Note: We use the train inputs here because that is the word we're trying to predict
            one_hot_labels = tf.one_hot(train_model_labels, vocabulary_size)
            regularized_vars = model.get_trained_variables()
            print('\n'.join([x.name for x in regularized_vars]))
            regularization_cost = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(x) for x in regularized_vars])
            print(one_hot_labels)
            cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=out, labels=train_model_labels
                )) + regularization_cost
            idxs = tf.argmax(out, axis=1)
            print("idxs shape", idxs.shape)
            print(train_model_labels.shape)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(idxs, tf.cast(train_model_labels, tf.int64)), tf.float32))
            print('accuracy tensor: {}'.format(accuracy))
            tf.summary.scalar('cost', cost)
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('optimizer'):
            cost_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    # Train Now
    with tf.Session(graph=graph) as session:
        print('full folderpath: ', full_folderpath)
        writer = tf.summary.FileWriter(run_dir, session.graph)
        init.run()
        print('Started from the bottom')

        average_cost = 0
        val_inputs, val_model_inputs, val_labels, val_model_labels = generate_batch(
            batch_size=len(val_ids),
            skip_window=skip_window,
            num_skips=NUM_SKIPS,
            data=data,
            data_indices=val_ids,
            model_context=model_context,
            rd=reversed_dictionary,
            lookahead=lookahead,
            sample=False,
        )
        all_train_inputs, all_train_model_inputs, all_train_labels, all_train_model_labels = generate_batch(
            batch_size=len(train_ids),
            skip_window=skip_window,
            num_skips=NUM_SKIPS,
            data=data,
            data_indices=train_ids,
            model_context=model_context,
            rd=reversed_dictionary,
            lookahead=lookahead,
            sample=False,
        )
        val_feed_dict = {
            train_model_labels: val_model_labels,
            train_model_inputs: val_model_inputs,
        }
        all_train_data_feed_dict = {
            train_model_labels: all_train_model_labels,
            train_model_inputs: all_train_model_inputs,
        }
        for step in range(1, num_steps+1):
            batch_inputs, model_inputs, batch_labels, model_labels = generate_batch(
                batch_size=batch_size,
                skip_window=skip_window,
                num_skips=NUM_SKIPS,
                data=data,
                data_indices=train_ids,
                model_context=model_context,
                rd=reversed_dictionary,
                lookahead=lookahead,
            )

            feed_dict = {
                train_model_labels: model_labels,
                train_model_inputs: model_inputs,
            }

            to_fetch = [cost_optimizer, merged, cost, accuracy, idxs, out]
            run_metadata = tf.RunMetadata()
            _, summary, cost_result, acc_result, preds, full_pred = session.run(
                to_fetch,
                feed_dict=feed_dict,
                run_metadata=run_metadata
            )

            average_cost += cost_result
            writer.add_summary(summary, step)

            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if step % 2000 == 0:
                if step > 0:

                    average_cost /= 2000
                print('Average cost at step ', step, ': -----------', average_cost)
                print('Accuracy at step ', step, ': ', acc_result * 100, '%')
                for i in range(len(preds)):
                    if i <= 10:
                        print("expected {}:{} predicted {}:{}".format(
                            model_labels[i],
                            reversed_dictionary[model_labels[i]],
                            preds[i],
                            reversed_dictionary[preds[i]]),
                        )
                    else:
                        break
                average_cost = 0
            # if step % 1000 == 0:
                # validation_accuracy = session.run(
                #     accuracy,
                #     feed_dict=val_feed_dict,
                # )
                # train_accuracy = session.run(
                #     accuracy,
                #     feed_dict=all_train_data_feed_dict,
                # )
                # print("Validation Accuracy at step{}: {}%".format(step, validation_accuracy * 100))
                # print("Train Accuracy at step{}: {}%".format(step, train_accuracy * 100))
            if step % 1000 == 0:
                evaluator = BatchSizeEvaluator(batch_size, session, train_model_labels, train_model_inputs, accuracy)
                validation_accuracy = evaluator.eval(
                    val_model_labels,
                    val_model_inputs,
                )
                train_accuracy = evaluator.eval(
                    all_train_model_labels,
                    all_train_model_inputs,
                )
                print('Validation Accuracy: ', validation_accuracy)
                print('Train Accuracy: ', train_accuracy)
        # Write corresponding labels for the embeddings.
        with open(os.path.join(run_dir, 'metadata.tsv'), 'w') as f:
            for i in range(vocabulary_size):
                f.write(reversed_dictionary[i] + '\n')

        # Get Final Accuracy Metrics
        evaluator = BatchSizeEvaluator(batch_size, session, train_model_labels, train_model_inputs, accuracy)
        validation_accuracy = evaluator.eval(
            val_model_labels,
            val_model_inputs,
        )
        train_accuracy = evaluator.eval(
            all_train_model_labels,
            all_train_model_inputs,
        )

        # Save the model for checkpoints.
        saver.save(session, os.path.join(run_dir, 'model.ckpt'))

    writer.close()
    return float(validation_accuracy) * 100, dict(
        validation_accuracy=float(validation_accuracy) * 100,
        train_accuracy=float(train_accuracy) * 100,
        average_cost=float(average_cost),
        run_dir=run_dir,
    )


class BatchSizeEvaluator:

    def __init__(self, batch_size, session, label_ph, inputs_ph, acc_tensor):
        self.batch_size = batch_size
        self.session = session
        self.label_ph = label_ph
        self.inputs_ph = inputs_ph
        self.acc_tensor = acc_tensor

    def eval(self, labels, inputs):
        acc = 0
        runs = 0
        for b in range(0, len(labels), self.batch_size):
            runs += 1
            fd = {
                self.label_ph: labels[b: b + self.batch_size],
                self.inputs_ph: inputs[b: b + self.batch_size],
            }
            acc += self.session.run(
                self.acc_tensor,
                feed_dict=fd,
            )
        return acc / runs * 100


def get_model(model_name, **kwargs):
    if model_name == 'lstm':
        return models.LSTMModel(**kwargs)
    if model_name == 'cnn':
        return models.CNNModel(**kwargs)
    if model_name == "split_cnn":
        return models.SplitCNN(**kwargs)
    raise Exception('Invalid Model Type: {}'.format(model_name))

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename, 'r') as f:
        data = f.read().split()
    return data

def tokenize(vocabulary, dictionary):
    """maps a dataset to a tokenization given a dicitonary"""
    data = list()
    unknown_count = 0
    for word in vocabulary:
        idx = dictionary.get(word + '\n', 0)
        if idx == 0:
            unknown_count += 1
        data.append(idx)
    print("Len data: {} --- unknown count: {}".format(len(data), unknown_count))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--pfile")  # parameters file
    parser.add_argument("--is_grid")
    args = parser.parse_args()
    if args.is_grid:
        run_grid_search(args, main)
    else:
        run_experiment(args, main)