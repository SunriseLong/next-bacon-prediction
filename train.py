# Same as notebook. Just quicker to inspect on GitHub

import re
import json
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import keras

bacon_path = keras.utils.get_file(
    "bacon-essays-92.txt",
    origin= "https://raw.githubusercontent.com/harrymqz/data_mining//master/data/NONFICTION/bacon-essays-92.txt")

bacon = open(bacon_path).read().lower()

bacon_clean = bacon.replace("\n", " ").replace(',', '')

bacon_tok = re.findall(r"[\w']+|[.,!?;]", bacon_clean[155:])

# create dictionary of word counts
word_dict = {}
for w in bacon_tok:
  if w in word_dict:
    word_dict[w] += 1
  else:
    word_dict[w] = 1

word_freq = sorted([(freq, w) for w, freq in word_dict.items()], reverse=True)
words = [w[1] for w in word_freq]

word_freq = word_freq[:1999]
index2word = {i: w[1] for i, w in enumerate(word_freq, 0)}
index2word[1999] = '<UNK>'
word2index = {w: i for i, w in index2word.items()}

# window over corpus and randomly sample phrases of 5-15 words
index = 0
X = []
y = []
for i in range(1, 2000):
  rv = np.random.randint(5,15)
  X.append(' '.join(bacon_tok[index: index+rv]))
  y.append(bacon_tok[index+rv])
  index+=(rv+1)

xtrain = np.array(X, dtype=object)[:, np.newaxis]

# OHE target
ytrain = list(map(lambda w: word2index[w] if w in word2index else 1999, y))
ytrain = np.eye(2000)[ytrain]

# ensure version 5. prev models are not keras serialization enabled
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embed = hub.KerasLayer(module_url, output_shape=(512,), input_shape=[], dtype=tf.string)

model = tf.keras.Sequential()
model.add(embed)
model.add(tf.keras.layers.Dense(2500, activation='relu'))
model.add(tf.keras.layers.Dense(2000, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xtrain, ytrain, epochs=20, batch_size=32)

from google.colab import files

model_path = "/tmp/model_artifacts"
model.save(model_path)
files.download("/tmp/model_artifacts/saved_model.pb")

# save word index to word mapping for inference
with open('/api/index2word.json', 'w') as fp:
    json.dump(index2word, fp)