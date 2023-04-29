#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

# Importing only MIT or Apache 2.0 licensed packages

import nltk
import re

import numpy as np
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
from PIL import Image as img
from .Visualizer import *


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
        self.dataframe_placeholder = st.empty()
        # self.model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
        # self.model = pipeline("sentiment-analysis")
        self.model = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyse(self):
        if type(self.user_input) is str:
            # specific_model = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
            sentiment = self.model(self.user_input)
            st.text(f"Sentiment :{sentiment[0]['label'].capitalize()}")
            st.text(f"Score: {sentiment[0]['score']}")
            image = img.open(os.path.join(os.getcwd(), "images","positive_smiley.jpg"))
            image1 = img.open(os.path.join(os.getcwd(),"images","Neutral.jpg"))
            image2 = img.open(os.path.join(os.getcwd(),"images","Negative.jpg"))

            if sentiment[0]["label"].capitalize() in["POSITIVE","POS", "Positive"]:
                st.image([image], width=60)
            elif sentiment[0]['label'] in ["NEGATIVE","NEG", "Negative"]:
                st.image([image2], width=60)
            else:
                st.image([image1], width=60)

    def analyse_dataset(self, count: int, df):
        # df = pd.read_csv(self.user_input)
        df = df["Comment"].head(count)
        word_count = 0
        df = pd.DataFrame(df, columns=["Comment"])
        df2 = pd.DataFrame(columns=["Comment", "Polarity", "score"])
        df["Polarity"] =None
        df["score"]= 0
        for i, row in df.iterrows():
            with st.spinner(f"analysed {i+1} comments"):
                polarity, score = self.get_sentiment(row.Comment)
                df["Polarity"].iloc[i] = polarity
                df["score"].iloc[i] = score
                new_data = {'Comment': row.Comment, 'Polarity': polarity, 'score': score}
                df2 = df2.append(new_data, ignore_index=True)
                self.dataframe_placeholder.dataframe(df2, width=2500)
                word_count += len(row.Comment.split())
        process = PostProcessor(df["score"].loc[df['Polarity'] == "Positive"])
        process.plot_sentiments("positive")
        process.plot_pie_chart(df["Polarity"])
        process1 = PostProcessor(df)
        process1.plot_word_cloud(word_count)

    def get_sentiment(self, input_comment):
        sentiment = self.model(input_comment)
        print(sentiment[0]["label"], sentiment[0]["score"])
        return sentiment[0]["label"], sentiment[0]["score"]


