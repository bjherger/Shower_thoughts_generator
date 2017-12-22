# Deep (Shower) Thought
## Teaching AI to generate posts to Reddit's r/Showerthoughts

**tl;dr:** TODO

Deep learning has drastically changed the way machines interact with human language. From 
[machine translation](https://research.google.com/pubs/pub45610.html) to 
[textbook writing](https://newatlas.com/writing-algorithm/25539/), Natural Language Processing (NLP) -- the branch of 
ML focused on human language models-- has gone from sci-fi to [readily available code](https://github.com/keras-team/keras/tree/master/examples#text--sequences-examples). 

Though I've had some previous experience with linear NLP models and word level deep learning models, I was interested 
in better understanding character level models. Generally, character level models look at a window of preceding 
characters, and try to infer the next character. Similar to repeatedly pressing auto-correct's top choice, this process 
can be repeated to generate a string of AI generated characters. 

TODO Image of generated sequence

Utilizing training data from [r/Showerthoughts](https://www.reddit.com/r/Showerthoughts/), and [starter code](github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)
from Keras, I built and trained a model that learned to generate new (and sometimes profound) shower thoughts 
 
## Data

[r/Showerthoughts](https://www.reddit.com/r/Showerthoughts/) is an online message board, to "share those miniature 
epiphanies you have" while in the shower. A few prime examples include:
 
 - `Every machine can be utilised as a smoke machine if it is used wrong enough.`
 - `It kinda makes sense that the target audience for fidget spinners lost interest in them so quickly`
 - `Google should make it so that looking up "Is Santa real?" With safe search on only gives yes answers.`
 - `Machine Learning is to Computers what Evolution is to Organisms.`

I scraped all posts for a 100 day period in 2017 utilizing Reddit's [PRAW](https://praw.readthedocs.io/en/latest/) Python 
API wrapper. Though I was mainly interested in the `titletext` field, a long list of other fields were available, 
including:

| variable     | type   |
|--------------|--------|
| author       | string |
| id           | string |
| name         | string |
| selftext     | string |
| title        | string |
| url          | string |
| downs        | int    |
| likes        | int    |
| num_comments | int    |
| score        | int    |
| ups          | int    |
| over_18      | bool   |
| spoiler      | bool   |

Once I had the data set, I performed a set of standard data transformations, including:

 - Converted the string to a list of characters
 - Replacing all illegal characters with a space. 
 - Lowercase-ing all characters
 - Converting text into an `X` array containing a fixed length arrays of characters, and a `y` array, containing the 
 next character. 
 
For example `It's 2017, any place that charges a convenience fee to pay bills online is just an as...` 
 would become the `X`, `y` pair: `['i', 't', ' ', 's', ' ', '2', '0', '1', '7', ' ', 'a', 'n', 'y', ' ', 'p', 'l', 'a', 'c', 'e', ' ', 't', 'h', 'a', 't', ' ', 'c', 'h', 'a', 'r', 'g', 'e', 's', ' ', 'a', ' ', 'c', 'o', 'n', 'v', 'e', 'n', 'i', 'e', 'n', 'c', 'e', ' ', 'f', 'e', 'e', ' ', 't', 'o', ' ', 'p', 'a', 'y', ' ', 'b', 'i', 'l', 'l', 's', ' ', 'o', 'n', 'l', 'i', 'n', 'e', ' ', 'i', 's', ' ', 'j', 'u', 's', 't', ' ', 'a', 'n', ' ', 'a']`, 
`s`.    

## Model

Data in hand, I built a model. Similar to the keras example code, I went with a Recurrent Neural Network (RNN), with 
Long Short Term Memory (LSTM) cells. Why this particular architecture choice works is beyond the scope of this post, but 
[this paper](https://arxiv.org/abs/1412.3555) covers it pretty well. 

In addition to the LSTM architecture, I chose to add a character embedding layer. Heuristically, there didn't seem to 
be much of a difference between one hot encoded and embedded inputs, but the embedding layers didn't greatly increase 
training time, and could allow for interesting further work. In particular, it would be interesting to look at character 
clustering and distances, similar to [Guo and Berkhahn](https://arxiv.org/abs/1604.06737).

Ultimately, the model looked something like: 

```python
x = keras.Input(..., dtype=dtype, name='char_input')
x = keras.Input(..., dtype=dtype, name='char_input')
x = Embedding(...,
    	name='char_embedding')(x)
x = LSTM(128, dropout=.2, recurrent_dropout=.2)(x)
x = Dense(..., activation='softmax')(x)

optimizer = RMSprop(lr=.001)

char_model = Model(sequence_input, x)
char_model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

Unfortunately, this character level model performed quite poorly. This is perhaps due to the variety in post content 
and writing styles, or the compounding effect of using predicted characters to infer additional characters. In the 
future, it would be interesting to look at predicting multiple characters at a time, or building a model that predicts 
words rather than characters.  

## Results

While this model struggled with the ephiphanies and profoundness of r/Showerthoughts, it was able to learn basic 
spelling, a complex (and unsurprisingly foul) vocabulary, and even basic grammar rules. Though the standard Nietzsche 
data set produces more intelligible results, this data set provided a more interesting challenge. 

If you're interested in the code to create the data set and train the LSTM model, check out the 
[repo](https://github.com/bjherger/Shower_thoughts_generator/). And the next time your in the shower, think about 
this: [We are giving AI a bunch of bad ideas with AI movies](https://www.reddit.com/r/Showerthoughts/comments/7dqoqu/we_are_giving_ai_a_bunch_of_bad_ideas_with_ai/?utm_term=df9c5226-c3e9-4742-beca-c1b8093b2948&utm_medium=search&utm_source=reddit&utm_name=Showerthoughts&utm_content=3)
