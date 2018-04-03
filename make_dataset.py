import string
import re
from static import contraction_map
from collections import namedtuple
from typing import List, Tuple, AnyStr
from datetime import datetime
import os
from random import shuffle
import argparse

RegexReplacement = namedtuple('RegexReplacement', ['rx', 'replacement'])

splitter = re.compile('[' + string.punctuation + string.whitespace + ']')

cleaned_data_path = './cleaned_data/'
DB_PATH = './db'
def generate_dataset(name, min_songs_for_artist, write_mode='w'):
    """

    :return:
    """
    datestamp = str(datetime.now().date()).replace('-', '')
    datapath = os.path.join(cleaned_data_path, datestamp)

    if not os.path.isdir(cleaned_data_path):
        os.makedirs(os.path.join(datapath))
    docs = _sample_lyrics(min_songs_for_artist)
    f = open(os.path.join(datapath, name), write_mode)
    regex_replacements = []
    for k in contraction_map:
        regex_replacements.append(RegexReplacement(re.compile("(?i)" + k), contraction_map[k]))
    print("Num Docs: {}".format(len(docs)))
    problems = 0
    problem_artists = set()
    for doc in docs:

        with open(os.path.join(DB_PATH, doc), 'r') as d:
            txt = d.read()
            cleaned, has_problem = _clean_data(regex_replacements, txt)
            if has_problem:
                problem_artists.add(doc.split('/')[0])
                problems += 1
                continue
            f.write(cleaned)
    print("num problems: {}".format(problems))
    f.close()
    with open(os.path.join(datapath, 'problem_artists'), 'w') as f:
        f.write('\n'.join(problem_artists))


def _sample_lyrics(min_songs_for_artist):
    """

    :return:
    """
    docs = []
    artists = os.listdir(DB_PATH)
    for a in artists:
        songs = os.listdir(os.path.join(DB_PATH, a))
        if len(songs) > min_songs_for_artist:
            docs.extend(map(
                lambda x: os.path.join(a, x), filter(
                    lambda x: 'remix' not in x and 'translation' not in x and x != 'urls.txt', songs
                ))
            )
    shuffle(docs)
    return docs


def _clean_data(regex_replacements: List[RegexReplacement], file_text):
    brackets = "\\[.*\\]"
    removed_bracket_notes = re.sub(brackets, "", file_text)
    if "https://" in file_text:

        return "", True

    txt = removed_bracket_notes
    for rr in regex_replacements:
        txt = rr.rx.sub(rr.replacement, txt)

    return ' '.join(filter(lambda x: x != '', splitter.split(txt))) + ' ', False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--write_mode")
    parser.add_argument("--min_songs_for_artist", type=int)
    args = parser.parse_args()
    generate_dataset(args.name, args.min_songs_for_artist, args.write_mode if args.write_mode else 'w')