# What's The Word

What's the word is a project attempting that attempts to
1. Build a language model
2. Leverage word prediction on the underlying language model.

## Dataset
What's The Word is trained on a lyrics dataset generated and scraped
from genius.com (`scrape` package). 

## Algorithms
After intial data preprocessing, we aim to build 
1. A language model using word embeddings 
2. A predictive network that can take in the context of a missing word
and predict the missing word. (At the moment a CNN is used)

## TODO
### Methods 
1. Learn the Embedding first
2. Learn the embedding with the prediction loss, but penalize the 
prediction loss less at the beginning
3. Use embeddings for labels rather than one hot (try both, 
weighting one hot more towards the beginning)

###  Parameters to tweak
1. Embedding Size (try smaller, then bigger)
2. Model Context (8 to each side?)

### Debugging
* If we zoom towards sparsity on the larger dataset, try again with a 
dataset exclusively from one artist
