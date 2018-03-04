import requests
import json
from bs4 import BeautifulSoup, Comment
import re
import os
import datetime
import math


lyric_segment_annotation = re.compile("\[.*\]")

base_url = "https://api.genius.com"

token = os.environ['GENIUS_TOKEN']
MAX_SONGS_PER_REQUEST = 50

f = open('./db/artist_maps/1519639336.json', 'r')
ARTIST_MAP = json.load(f)
f.close()


def download_lyrics(artist, number_of_songs=20):
    artist_id = ARTIST_MAP[artist]
    page = 100
    songs_per_request = min(number_of_songs, MAX_SONGS_PER_REQUEST)
    while number_of_songs > 0:
        songs = _get_songs_for_artist(artist_id, songs_per_request, page)
        page += 1
        number_of_songs -= songs_per_request
        _write_songs(songs, artist)


def _write_songs(songs, artist):
    print("writing to file")
    artist_path = './db/{}'.format(artist)
    os.makedirs(artist_path, exist_ok=True)
    for idx, (title, lyrics) in enumerate(songs):
        modified_title = title.replace('/', '_').replace(' ', '_')
        with open(os.path.join(artist_path + '/{}_{}.txt'.format(modified_title, idx)), 'w') as g:
            g.write(lyrics)
    return


def _get_songs_for_artist(artist_id, number_of_songs, page):
    """
    :param artist_id: Genius artist ID
    :param number_of_songs: Number of songs to request
    :return: List of tuples with the title and the lyrics, each as strings
    """
    titles_and_lyrics = []

    response = requests.request(
        "GET",
        "{}/artists/{}/songs".format(base_url, artist_id),
        headers={"Authorization": "Bearer {}".format(token)},
        params={"per_page": number_of_songs, "page": page}
    )
    if response.status_code != 200:
        raise Exception("Bad Response: \nCode: {}\n Content:{}".format(response.status_code, response.content))

    response_content = json.loads(response.content)
    songs = response_content['response']['songs']

    for song in songs:
        print('getting song for url: {}'.format(song['url']))
        titles_and_lyrics.append((song['title'], _get_lyrics_for_url(song['url'])))

    return titles_and_lyrics


def make_artist_map(start=1, end=20000):
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


    #TODO MAKE MAP AND WRITE TO DISK


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
