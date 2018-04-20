import string
import re
from static import contraction_map
from collections import namedtuple
from typing import List
from datetime import datetime
import os
from random import shuffle
import argparse

RegexReplacement = namedtuple('RegexReplacement', ['rx', 'replacement'])

SPLITTER = re.compile('[' + string.punctuation + string.whitespace + ']')
NORMAL_LETTERS = re.compile('^[0-9a-zA-Z]+$')

cleaned_data_path = './cleaned_data/'
DB_PATH = './db'
def generate_dataset(name, min_songs_for_artist, write_mode='w'):
    """
    Generates a dataset for all artists who have more than the `min_songs_for_artist`

    Writes the songs to disk under `cleaned_data/<datestamp/name`
    """
    datestamp = str(datetime.now().date()).replace('-', '')
    datapath = os.path.join(cleaned_data_path, datestamp)
    if not os.path.isdir(datapath):
        os.makedirs(datapath)

    # Fetch all the filenames that we will we read into our large dataset file
    docs = _sample_lyrics(min_songs_for_artist)
    f = open(os.path.join(datapath, name), write_mode)
    regex_replacements = []

    # Load the replacements mapping into compiled regexes
    for k in contraction_map:
        regex_replacements.append(RegexReplacement(re.compile("(?i)" + k), contraction_map[k]))
    print("Num Docs: {}".format(len(docs)))

    # When downloading the data, rapgenius decided to not give me lyrics but rather urls when requesting
    # a song. Catch these issues and see which artists have no lyrics.
    problems = 0
    problem_artists = set()
    noproblem_artistis = set()
    for doc in docs:
        with open(os.path.join(DB_PATH, doc), 'r') as d:
            txt = d.read()
            cleaned, has_problem = _clean_data(regex_replacements, txt)
            if has_problem:
                problem_artists.add(doc.split('/')[0])
                problems += 1
                continue
            noproblem_artistis.add(doc.split('/')[0])
            f.write(cleaned)
    print("num problems: {}".format(problems))
    f.close()

    # Note which artists we would need to redownload lyrics for.
    with open(os.path.join(datapath, 'problem_artists'), 'w') as f:
        f.write('\n'.join(problem_artists.difference(noproblem_artistis)))


def _sample_lyrics(min_songs_for_artist):
    """
    Pick which songs to take lyrics from.

    :return: list of filepaths relative to DB_PATH as root
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
    # Shuffle the list of files so that we can partition it easily into training/testing sets without
    # biasing a set to a given artist
    shuffle(docs)
    return docs


def _clean_data(regex_replacements: List[RegexReplacement], file_text):
    """
    Cleans up the data by replacing contractions and other slang, as well as cleaning up
    genius.com meta-notes like "[Verse 1]" etc.
    :param regex_replacements:
    :param file_text:
    :return: Cleaned `file_text` as a string
    """
    brackets = "\\[.*\\]"
    removed_bracket_notes = re.sub(brackets, "", file_text)
    if "https://" in file_text or len(file_text) == 0:

        return "", True

    txt = removed_bracket_notes
    for rr in regex_replacements:
        txt = rr.rx.sub(rr.replacement, txt)
    words = SPLITTER.split(txt.lower())
    filtered_words = filter(lambda x: NORMAL_LETTERS.match(x), words)

    return ' '.join(filtered_words) + ' ', False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--write_mode")
    parser.add_argument("--min_songs_for_artist", type=int)
    args = parser.parse_args()
    generate_dataset(args.name, args.min_songs_for_artist, args.write_mode if args.write_mode else 'w')