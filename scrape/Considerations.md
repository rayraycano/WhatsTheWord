# Considerations Made while Constructing the Dataset

## Dataset Curation

1. Ping Genius's API with `artist_id`s sent sequentially to build out a mapping of artist name -> `artist_id`
2. Select which artist's we want to include in the study.
3. Get all song lyric URLs for this artist via Genius API
4. Download lyrics for songs we want requesting the song URL and scraping the page sites.

There's a couple ways we can go about choosing our artists in the study

1. Random Sample
2. Top 100 Hip Hop Artists
3. Top 100 Hip Hop Artists + Top 100 Artists in general
4. All Artists

I think `1` is a decent first approach. The failure mode here is that our dataset doesn't represent the information that
the majority of the population consumes when it listens to rap. That inspires methods 2 and 3. Method 3 exists to get
variation in the database to perhaps provide more alternatives for words we would like to provide semantic neighbors for,
namely the n-word. The last method...is infeasible.

## Filtering songs

Songs were gathered by pulling all songs that Genius.com had tied to an artist. These URLs are then pulled and the
lyrics are downloaded. When looking through the song urls though, there is some unfamiliar content. It seems that if
anyone publishes lyrics and links to a given artist, that song becomes tied to the artist (for example, see https://genius.com/Fried-rice-240-hours-lyrics).

In addition, there are translations and remixes floating around. We could accept these as is, or, we could try to ignore them.
Another thing we could look at is the distribution of the number of songs by artist. Is it gonna matter that much if we do
end up with one of these edge cases?

- banned title words
    - translation
    - remix

## Data Cleansing

This pertains to treating the actual lyrics
- Remove "[.*]" from data
- Lowercase
- Remove Punctuation
- Stemming
    - Traditional Stemmers don't seem to take into account the slang conjugations we see in rap
    - One option here is to not stem at all. We could collect enough data that slang words pick up enough meaning.
    - From a context perspective, they should not matter too much. We can take the problem of replacing the n-word in a
    in a given context and use the same nearest neighbor/highest probability approach to find synonyms to slang

