import requests
import json
from bs4 import BeautifulSoup, Comment
import re
import os

lyric_segment_annotation = re.compile("\[.*\]")

base_url = "https://api.genius.com"

token = os.environ['GENIUS_TOKEN']

def download_lyrics(artist):
    artist_id = 16775   #TODO: Make name -> ID Mapping
    songs = _get_songs_for_artist(artist_id)
    _write_songs(songs, artist)

def _write_songs(songs, artist):
    artist_path = './db/{}'.format(artist)
    os.makedirs(artist_path, exist_ok=True)
    for idx, song in enumerate(songs):
        with open(os.path.join(artist_path + '/{}.txt'.format(idx)), 'w') as f:
            f.write(song)
    return


def _get_songs_for_artist(artist_id, number_of_songs=20):
    """
    :param artist_id:
    :param number_of_songs:
    :return:
    """
    response = requests.request(
        "GET",
        "{}/artists/{}/songs".format(base_url, artist_id),
        headers={"Authorization": "Bearer {}".format(token)},
        params={"per_page": number_of_songs}
    )
    if response.status_code != 200:
        raise Exception("Bad Response: \nCode: {}\n Content:{}".format(response.status_code, response.content))

    response_content = json.loads(response.content)
    songs = response_content['response']['songs']
    lyrics = []
    for song in songs:
        print('getting song for url: {}'.format(song['url']))
        lyrics.append(_get_lyrics_for_url(song['url']))
    return lyrics




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
    for c in lyrics_div.descendants:
        if c.string is not None:
            lines.append(c.string.strip())
    filtered_lines = filter(lambda text: text != '' and lyric_segment_annotation.match(text) is None, lines)
    return '\n'.join(list(filtered_lines))
