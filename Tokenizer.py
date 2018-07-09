import nltk
import string
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


class Tokenizer:
    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens = [i for i in tokens if (i not in string.punctuation)]
        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
        tokens = [i for i in tokens if (i not in stop_words)]
        tokens = [i.replace("«", "").replace("»", "") for i in tokens]
        return tokens

    @staticmethod
    def tokenize_and_vectorize(vectorizer, data):
        max_tokens = vectorizer.syn0.shape[0]
        tokens = Tokenizer.tokenize(data)
        tokens = [vectorizer.vocab[token].index if token in vectorizer.vocab else max_tokens for token in tokens]
        return tokens
