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
            :param : tweet column
    """

    # Defining a constructor for the class
    def __init__(self, data: pd.dataframe):
        self.data = data

    # Step 1: Text cleaning
    @staticmethod
    def clean_text(text):
        """
        :param text: Individual comment in the data
        :return: cleaned comments
        """
        text = text.lower()
        text = nltk.re.sub(r'\d+', '', text)  # remove digits
        text = nltk.re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = nltk.re.sub(r'http?\S+', ' ', text)  # remove extra spaces and url
        return text

    # Step 2: Tokenization
    @staticmethod
    def tokenize(text: str) -> list:
        """
        Break the text into individual words or phrases (n-grams) called tokens
        :param text: Individual comment in the data
        :return: words returned in a list
        """
        tokens = word_tokenize(text)
        return tokens

    # Step 3: Lemmatization
    @staticmethod
    def lemmatize(self, tokens: list) -> list:
        lemmatizer = WordNetLemmatizer()
        stopwords_list = stopwords.words('english')
        normalized_tokens = []
        for token in tokens:
            if token not in stopwords_list:
                normalized_token = lemmatizer.lemmatize(token)
                normalized_tokens.append(normalized_token)
        return normalized_tokens

    # Step 4: Feature selection
    @staticmethod
    def select_features(self, x):
        x_tfidf = TfidfVectorizer()
        return x_tfidf

    def preprocess(self):
        data = self.data
        data['Comments'] = data['Comments'].apply(self.clean_text)
        data['tokens'] = data['Comments'].apply(self.tokenize)
        data['lemmatized_tokens'] = data['tokens'].apply(self.lemmatize)
        x = data['lemmatized_tokens']
        sentiments = []
        blob = TextBlob(x)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
        df = pd.DataFrame({'sentiment': sentiments})
        data["polarity"] = df
        y = data['polarity']
        x = self.select_features(x)
        return x, y
