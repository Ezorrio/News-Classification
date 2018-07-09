from __future__ import print_function

import csv

import gensim
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed, Dropout, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, save_model, load_model
from tqdm import tqdm

from Tokenizer import Tokenizer


class RCNN:
    def __init__(self, vectorizer, categories, restore_file=None):
        categories.sort()
        self.vectorizer = vectorizer
        self.categories = categories
        max_tokens = vectorizer.syn0.shape[0]
        embedding_dim = vectorizer.syn0.shape[1]
        hidden_dim_1 = 200
        hidden_dim_2 = 100
        num_class = len(categories)

        if restore_file is None:
            # create model for training
            embeddings = np.zeros((vectorizer.syn0.shape[0] + 1, vectorizer.syn0.shape[1]), dtype="float32")
            embeddings[:vectorizer.syn0.shape[0]] = vectorizer.syn0

            document = Input(shape=(None,), dtype="int32")
            left_context = Input(shape=(None,), dtype="int32")
            right_context = Input(shape=(None,), dtype="int32")

            embedder = Embedding(max_tokens + 1, embedding_dim, weights=[embeddings], trainable=False)
            doc_embedding = embedder(document)
            l_embedding = embedder(left_context)
            r_embedding = embedder(right_context)

            forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)
            backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)

            backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
            together = concatenate([forward, doc_embedding, backward], axis=2)

            semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)
            pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)

            output = Dense(num_class, input_dim=hidden_dim_2, activation="softmax")(pool_rnn)

            self.model = Model(inputs=[document, left_context, right_context], outputs=output)
            self.model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            self.restoreModel(restore_file)

    def trainModelFromFile(self, file_name):
        max_tokens = Vectors.getMaxToken(self.vectorizer)
        num_classes = len(self.categories)
        with open(file_name, encoding="utf-8") as f:
            f.readline()
            reader = csv.reader(f)
            for line in tqdm(reader):
                title = line[0]
                category = line[1]
                data = line[2]
                text = title + "\n" + data

                tokens = Tokenizer.tokenize_and_vectorize(text)
                doc_as_array = np.array([tokens])
                # We shift the document to the right to obtain the left-side contexts.
                left_context_as_array = np.array([[max_tokens] + tokens[:-1]])
                # We shift the document to the left to obtain the right-side contexts.
                right_context_as_array = np.array([tokens[1:] + [max_tokens]])

                target = np.array([num_classes * [0]])
                # find corresponding position
                # print(category)
                index = self.categories.index(category)
                target[0][index] = 1
                self.model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, verbose=0)

    def trainModel(self, x_train, y_train):
        max_tokens = Vectors.getMaxToken(self.vectorizer)
        num_classes = len(self.categories)
        for index, row in tqdm(x_train.iterrows()):
            title = row['title']
            data = row['text']
            text = title + "\n" + data

            category = y_train.get(index)
            tokens = Tokenizer.tokenize_and_vectorize(self.vectorizer, text)

            doc_as_array = np.array([tokens])
            left_context_as_array = np.array([[max_tokens] + tokens[:-1]])
            right_context_as_array = np.array([tokens[1:] + [max_tokens]])

            target = np.array([num_classes * [0]])
            index = self.categories.index(category)
            target[0][index] = 1
            self.model.train_on_batch([doc_as_array, left_context_as_array, right_context_as_array], target)

    def predictFromFile(self):
        with open('data/test.txt', 'r', encoding="utf-8") as myfile:
            test = myfile.read()
            print(test)
            print(".............")
            print(self.predictModel(test))

    def predictModel(self, text):
        max_tokens = Vectors.getMaxToken(self.vectorizer)
        tokens = Tokenizer.tokenize_and_vectorize(self.vectorizer, text)
        central = np.array([tokens])
        left = np.array([[max_tokens] + tokens[:-1]])
        right = np.array([tokens[1:] + [max_tokens]])
        weights = self.model.predict([central, left, right])
        max_weight_index = np.argmax(weights)
        return self.categories[max_weight_index]

    def evaluateModel(self, x_test, y_test, max_count=None):
        correct_count = 0
        incorrect_count = 0
        for index, row in x_test.iterrows():
            title = row['title']
            category = y_test.get(index)
            data = row['text']
            text = title + "\n" + data
            if self.predictModel(text) == category:
                correct_count += 1
            else:
                incorrect_count += 1
                print("Incorrect", "Title " + title, "Expected " + self.predictModel(text), "Real " + category)
            if (max_count is not None) and (max_count <= (correct_count + incorrect_count)):
                break
        print("Correct count", correct_count)
        print("Incorrect count", incorrect_count)
        print("Accuracy", correct_count / (correct_count + incorrect_count), "%")

    # must finish on .HDF5
    def saveModel(self, file_name):
        self.model.save("result/" + file_name)

    def restoreModel(self, file_name):
        self.model = load_model("result/" + file_name)

    # @staticmethod
    # def getRNNModel():
    #     model_conv = Sequential()
    #     model_conv.add(Embedding(vocabulary_size, 100, input_length=50))
    #     model_conv.add(Dropout(0.2))
    #     model_conv.add(Conv1D(64, 5, activation='relu'))
    #     model_conv.add(MaxPooling1D(pool_size=4))
    #     model_conv.add(LSTM(100))
    #     model_conv.add(Dense(1, activation='sigmoid'))
    #     model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     return model_conv


class Vectors:
    @staticmethod
    def load(file_name):
        return gensim.models.KeyedVectors.load_word2vec_format(file_name)

    @staticmethod
    def getWord2Vec():
        return Vectors.load("vectors/word2vec/word2vec_920000.vec")

    @staticmethod
    def getFastText():
        return Vectors.load("vectors/fasttext/wiki_half.vec")

    @staticmethod
    def getMaxToken(vectorizer):
        return vectorizer.syn0.shape[0]
