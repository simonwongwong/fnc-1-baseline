import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

import numpy as np
import pandas as pd
from utils.dataset import DataSet
from utils.generate_test_splits import generate_hold_out_split, read_ids

d = DataSet()
generate_hold_out_split(d)
trainID = set(read_ids("training_ids.txt", "splits"))
valID = set(read_ids("hold_out_ids.txt", "splits"))

MAX_SENT_PER_ART = 20
MAX_SENT_LEN = 30
MAX_VOCAB = 20000

train_stances = [stance for stance in d.stances if stance['Body ID'] in trainID]
train_headlines = [stance['Headline'] for stance in train_stances]
train_labels = [stance['Stance'] for stance in train_stances]
train_body = [d.articles[stance['Body ID']] for stance in train_stances]

val_stances = [stance for stance in d.stances if stance['Body ID'] in valID]
val_headlines = [stance['Headline'] for stance in val_stances]
val_labels = [stance['Stance'] for stance in val_stances]
val_body = [d.articles[stance['Body ID']] for stance in val_stances]

import nltk
nltk.download('punkt')

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(train_body + train_headlines + val_body + val_headlines)

from nltk import tokenize

sent_tok_art = []
for article in train_body:
    sent_tok_art.append(tokenize.sent_tokenize(article))

vsent_tok_art = []
for article in val_body:
    vsent_tok_art.append(tokenize.sent_tokenize(article))

X_train_body = np.zeros((len(train_stances), MAX_SENT_PER_ART, MAX_SENT_LEN), dtype='int32')

for i, article in enumerate(sent_tok_art):
    for j, sentence in enumerate(article[:MAX_SENT_PER_ART]):
        words = text_to_word_sequence(sentence)
        for k, word in enumerate(words[:MAX_SENT_LEN]):
            X_train_body[i][j][k] = tokenizer.word_index[word]

X_train_head = np.zeros((len(train_stances), MAX_SENT_LEN), dtype='int32')

for i, headline in enumerate(train_headlines):
    words = text_to_word_sequence(headline)
    for j, word in enumerate(words[:MAX_SENT_LEN]):
        X_train_head[i][j] = tokenizer.word_index[word]

X_val_body = np.zeros((len(val_stances), MAX_SENT_PER_ART, MAX_SENT_LEN), dtype='int32')

for i, article in enumerate(vsent_tok_art):
    for j, sentence in enumerate(article[:MAX_SENT_PER_ART]):
        words = text_to_word_sequence(sentence)
        for k, word in enumerate(words[:MAX_SENT_LEN]):
            X_val_body[i][j][k] = tokenizer.word_index[word]

X_val_head = np.zeros((len(val_stances), MAX_SENT_LEN), dtype='int32')

for i, headline in enumerate(val_headlines):
    words = text_to_word_sequence(headline)
    for j, word in enumerate(words[:MAX_SENT_LEN]):
        X_val_head[i][j] = tokenizer.word_index[word]

targets = pd.Series(train_labels)
one_hot = pd.get_dummies(targets,sparse = True)
one_hot_labels = np.asarray(one_hot)
y_train = one_hot_labels

targets = pd.Series(val_labels)
one_hot = pd.get_dummies(targets,sparse = True)
one_hot_labels = np.asarray(one_hot)
y_val = one_hot_labels

vocab_size = len(tokenizer.word_index)
embedding_matrix = np.zeros((vocab_size+1, 300))

for word, i in tokenizer.word_index.items():
    try:
        v = wv[word]
        embedding_matrix[i] = v
    except KeyError:
        pass

from keras.models import Sequential
from keras.layers import Dense,LSTM, TimeDistributed, Activation
from keras.layers import Flatten, Permute, merge, Input
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Input,Dense,multiply,concatenate,Dropout
from keras.layers import GRU, Bidirectional

hidden_size = 300

sentence_input = Input(shape=(MAX_SENT_LEN,),dtype='int32')

embedded_sequences = Embedding(output_dim = hidden_size, input_dim = vocab_size+1, input_length=MAX_SENT_LEN)(sentence_input)

l_LSTM = Bidirectional(LSTM(hidden_size,return_sequences = True))(embedded_sequences)
l_dense = TimeDistributed(Dense(hidden_size))(l_LSTM)
l_dense = Flatten()(l_dense)
sentEncoder = Model(sentence_input,l_dense)

body_input = Input(shape=(MAX_SENT_PER_ART,MAX_SENT_LEN,),dtype = 'int32')

body_encoder = TimeDistributed(sentEncoder)(body_input)

l_LSTM_sent = Bidirectional(LSTM(hidden_size,return_sequences=True))(body_encoder)
l_dense_sent = TimeDistributed(Dense(hidden_size))(l_LSTM_sent)
l_dense_sent = Flatten()(l_dense_sent)

heading_input = Input(shape = (MAX_SENT_LEN, ),dtype = 'int32')
heading_embedded_sequences = Embedding(output_dim=hidden_size, input_dim=vocab_size+1, \
                                       input_length = (MAX_SENT_LEN,), \
                                      weights = [embedding_matrix])(heading_input)
h_dense = Dense(hidden_size,activation='relu')(heading_embedded_sequences)
h_flatten = Flatten()(h_dense)
article_output = concatenate([l_dense_sent,h_flatten],name = 'concatenate_heading')

news_vestor = Dense(hidden_size,activation = 'relu')(article_output)
preds = Dense(4,activation = 'softmax')(news_vestor)
model = Model([body_input,heading_input],[preds])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

model.fit([X_train_body,X_train_head],[y_train], validation_data=([X_val_body,X_val_head],[y_val]), epochs=10 , batch_size=100)