import requests
import json
from bs4 import BeautifulSoup, Comment
import re
import os
import datetime
import math
from numpy import random


lyric_segment_annotation = re.compile("\[.*\]")

base_url = "https://api.genius.com"

token = os.environ['GENIUS_TOKEN']
MAX_SONGS_PER_REQUEST = 50

f = open('./db/artist_maps/artist_map.json', 'r')
ARTIST_MAP = json.load(f)
f.close()

# edmund is cool
def get_all_urls(min_id=0):
    for artist in filter(lambda x: ARTIST_MAP[x] > min_id, ARTIST_MAP.keys()):
        download_lyrics(artist, float('inf'), save_urls_only=True)


def download_lyrics_from_urls(sample_rate=0.2, start=1):

    qualified_artists = filter(lambda x: start < ARTIST_MAP[x] < 400, ARTIST_MAP.keys())
    for artist in qualified_artists:
        # TODO: Move artist_map out of db
        if artist == 'artist_map':
            continue
        print('getting lyrics for artist {}'.format(artist))
        with open('./db/{}/urls.txt'.format(artist), 'r') as f:
            urls = f.read().split('\n')
        num_samples = int(math.ceil(len(urls) * sample_rate))
        sampled_songs = random.choice(urls, num_samples, False)
        titles_and_urls = [(url.replace('https://genius.com/', ''), _get_lyrics_for_url(url)) for url in sampled_songs]
        print("Writing for artist id: {}".format(ARTIST_MAP[artist]))
        _write_songs(titles_and_urls, artist)


def download_lyrics(artist, number_of_songs=20, save_urls_only=False):
    print('here')
    artist_id = ARTIST_MAP[artist]
    page = 1
    songs_per_request = min(number_of_songs, MAX_SONGS_PER_REQUEST)
    all_urls = []
    while number_of_songs > 0:
        titles_and_lyrics = []
        urls_and_titles, err = _get_song_urls_for_artist(artist_id, songs_per_request, page)
        if len(urls_and_titles) == 0:
            break
        page += 1
        number_of_songs -= songs_per_request
        all_urls.extend([x[0] for x in urls_and_titles])
        if not save_urls_only:
            for link, title in urls_and_titles:
                titles_and_lyrics.append((title, _get_lyrics_for_url(link)))
            _write_songs(titles_and_lyrics, artist)
    artist_name = artist.replace('/', '_')
    if not os.path.isdir('./db/{}'.format(artist_name)):
        os.mkdir('./db/{}'.format(artist_name))
    with open('./db/{}/urls.txt'.format(artist_name), 'w') as g:
        g.write('\n'.join(all_urls))

def _write_songs(songs, artist):
    print("writing to file")
    artist_path = './db/{}'.format(artist)
    os.makedirs(artist_path, exist_ok=True)
    for idx, (title, lyrics) in enumerate(songs):
        modified_title = title.replace('/', '_').replace(' ', '_')
        with open(os.path.join(artist_path + '/{}_{}.txt'.format(modified_title, idx)), 'w') as g:
            g.write(lyrics)
    return


def _get_song_urls_for_artist(artist_id, number_of_songs, page):
    """
    Get a list of song URLs for a given artist
    :param artist_id: Genius artist ID
    :param number_of_songs: Number of songs to request
    :param page: Genius pagination number
    :return: List of tuples with the title and the lyrics, each as strings
    """
    print("Fetching {} songs from page {} for artist_id: {}".format(number_of_songs, page, artist_id))
    response = requests.request(
        "GET",
        "{}/artists/{}/songs".format(base_url, artist_id),
        headers={"Authorization": "Bearer {}".format(token)},
        params={"per_page": number_of_songs, "page": page}
    )
    if response.status_code != 200:
        # raise Exception("Bad Response: \nCode: {}\n Content:{}".format(response.status_code, response.content))
        return [], True

    response_content = json.loads(response.content)
    songs = response_content['response']['songs']
    print("number of songs fetched: {}".format(len(songs)))
    return [(song['url'], song['title']) for song in songs], False


def make_artist_map(start=1, end=20000):
    """
    Recreate the artist mapping for Genius
    :param start: ID to start at
    :param end: ID to end at
    :return: None
    """

    map = {}
    map_path = 'db/artist_maps/{}.json'.format(math.floor(datetime.datetime.now().timestamp()))
    os.makedirs('db/artist_maps', exist_ok=True)
    for i in range(start, end):
        name, err = _get_artist_name(i)
        if err is not None:
            continue
        map[name] = i
        if i % 100:
            with open(map_path, 'w') as f:
                f.write(json.dumps(map, indent=4, separators=[',', ':']))


def _get_artist_name(artist_id):
    response = requests.request(
        "GET",
        "{}/artists/{}".format(base_url, artist_id),
        headers={"Authorization": "Bearer {}".format(token)},
    )
    response_content = json.loads(response.content)
    content = response_content.get('response')

    if content is None:
        return None, 'no response param'
    maybe_artist = content['artist']
    if maybe_artist is None:
        return None, 'no artist param'
    return maybe_artist['name'], None


def _get_lyrics_for_url(url):
    web_request = requests.request(
        "GET",
        url,
    )
    soup = BeautifulSoup(web_request.content, 'html.parser')
    lyrics_div = soup.find('div', class_='lyrics')
    for comment_element in lyrics_div(text=lambda text: isinstance(text, Comment)):
        comment_element.extract()
    lines = []
    last = -1
    for c in lyrics_div.descendants:
        if c.string is not None and last != c.previous_sibling:
            lines.append(c.string.strip())
            # print(c.tag)
            last = c

    filtered_lines = filter(lambda text: text != '', lines)
    return '\n'.join(list(filtered_lines))
