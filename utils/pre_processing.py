#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing only MIT or Apache 2.0 licensed packages

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from textblob import TextBlob
import re
import urlextract
from sklearn.metrics import classification_report


class Preprocessor:

    """
    Class representing basic Preprocessing steps for Sentiment Analysis

        Attributes:
            max_features : Count of features to be extracted from text during vectorization
            test_size : Splitting factor for train-test split data
            random_state : sample seed size for shuffling
    """

    # Defining a constructor for the class
    def __init__(self, max_features=1000, test_size=0.2, random_state=42):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = None

    # Step 1: Text cleaning
    def clean_text(self, text):
        """
        :param text: Individual comment in the data
        :return: cleaned comments
        """
        text = text.lower()
        text = nltk.re.sub(r'\d+', '', text)  # remove digits
        text = nltk.re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = nltk.re.sub(r'https?\S+', ' ', text)  # remove extra spaces and urls
        return text

    # Step 2: Remove emoticons
    def remove_emoticons(self, text):
        emoticon_pattern = re.compile("["
                                      u"\U0001F600-\U0001F64F"  # emoticons
                                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                      u"\U00002500-\U00002BEF"  # chinese char
                                      u"\U00002702-\U000027B0"
                                      u"\U00002702-\U000027B0"
                                      u"\U000024C2-\U0001F251"
                                      u"\U0001f926-\U0001f937"
                                      u"\U00010000-\U0010ffff"
                                      u"\u2640-\u2642"
                                      u"\u2600-\u2B55"
                                      u"\u200d"
                                      u"\u23cf"
                                      u"\u23e9"
                                      u"\u231a"
                                      u"\ufe0f"  # dingbats
                                      u"\u3030"
                                      "]+", flags=re.UNICODE)
        return emoticon_pattern.sub(r'', text)

    # Step 3: Tokenization
    def tokenize(self, text: str) -> list:
        """
        Break the text into individual words or phrases (n-grams) called tokens
        :param text: Individual comment in the data
        :return: words returned in a list
        """
        tokens = word_tokenize(text)
        return tokens

    # Step 4: Normalization
    def normalize(self, tokens) -> list:
        lemmatizer = WordNetLemmatizer()
        stopwords_list = stopwords.words('english')
        normalized_tokens = []
        for token in tokens:
            if token not in stopwords_list:
                normalized_token = lemmatizer.lemmatize(token)
                normalized_tokens.append(normalized_token)
        return normalized_tokens

    # Step 5: Stemming/Lemmatization
    def stem_or_lemmatize(self, tokens) -> list:
        lemmatizer = WordNetLemmatizer()
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = lemmatizer.lemmatize(token)
            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    # Step 6: Feature selection
    def select_features(self, X, y):
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        X = self.vectorizer.fit_transform(X)
        return X, y

    # Step 7: Vectorization of train and test data (to be included in the code)
    def vectorize(self, X_train, X_test):
        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        return X_train, X_test

    # Step 8: Splitting data
    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    # Single Function to preprocess data (calls internal Class methods)

    # Have to be modified for different sources , but different class objects can be created, hence modification might not be needed.

    # step to preprocess data automatically when the instance is called.
    def __call__(self, data):
        data['Comments'] = data['Comments'].apply(self.clean_text)
        data["Comments"] = data["Comments"].apply(self.remove_emoticons)
        data['tokens'] = data['Comments'].apply(self.tokenize)
        data['normalized_tokens'] = data['tokens'].apply(self.normalize)
        data['stemmed_tokens'] = data['normalized_tokens'].apply(self.stem_or_lemmatize)
        X = data['stemmed_tokens']
        sentiments = []
        blob = TextBlob(X)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
        df = pd.DataFrame({'sentiment': sentiments})
        data["polarity"] = df
        y = data['polarity']
        X, y = self.select_features(X, y)
        return X, y
