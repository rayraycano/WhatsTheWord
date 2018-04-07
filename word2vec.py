import argparse
import tensorflow as tf
import collections

def main(args):
    vocabulary = read_data(args.filename)
    print('Data size', len(vocabulary))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000
    data, count, dictionary, recersed_dictionary = build_dataset(vocabulary, vocabulary_size)


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename, 'r') as f:
        data = f.read().split()
    return data

def build_dataset(words, n_words):
    """
    Process raw inputs into a dataset.
    Taken from "https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/word2vec"
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

    args = parser.parse_args()
    main(args)
