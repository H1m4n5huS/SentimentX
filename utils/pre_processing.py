#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing only MIT or Apache 2.0 licensed packages

import nltk
import re
import unicodedata
import streamlit as st
import pandas as pd
import urlextract
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from tqdm import tqdm
from transformers import pipeline


class Preprocessor:

    """
    Class representing basic Preprocessing steps for Sentiment Analysis

        Attributes:
            :param : tweet column
    """

    # Defining a constructor for the class
    def __init__(self, data: pd.DataFrame):
        self.data = data

    # Step 1: Text cleaning
    @staticmethod
    def clean_text(text):
        """
        :param text: Individual comment in the data
        :return: cleaned comments
        """
        text = text.lower()  # converting to lowercase
        text = nltk.re.sub(r'\d+', '', text)  # remove digits
        text = nltk.re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = nltk.re.sub(r'http?\S+', ' ', text)  # remove extra spaces and url
        text = nltk.re.sub(r'https?\S+', ' ', text)  # remove secured urls
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

    # Step 4:
    def remove_non_ascii(self, tokens: list):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for token in tokens:
            new_word = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def preprocess(self, data):
        data = self.data
        # use tqdm to track progress
        tqdm.pandas()
        progress_bar = st.progress(0)
        data['Comments'] = data['Comments'].progress_apply(self.clean_text)
        data['tokens'] = data['Comments'].apply(self.tokenize)
        data['lemmatized_tokens'] = data['tokens'].apply(self.lemmatize)
        x = data['lemmatized_tokens']
        sentiments = []
        blob = TextBlob(x)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
        df = pd.DataFrame({'sentiment': sentiments})
        # self.data["polarity"] = df
        y = data['polarity']
        # x = self.select_features(x)
        # st.write(self.data.polarity)
        print(y)


class SentimentAnalyser:
    def __init__(self, user_input):
        self.user_input = user_input

    def analyse(self):
        if type(self.user_input) is str:
            # specific_model = pipeline('sentiment-analysis',  model="nlptown/bert-base-multilingual-uncased-sentiment")
            specific_model = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
            # specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

            sentiment = specific_model(self.user_input)
            print(sentiment[0])
            print(sentiment)
            st.text(sentiment)
