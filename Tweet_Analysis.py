# import tweepy as tw
import os
import time
import sys
import streamlit as st
import PIL
from PIL import Image as img
import pandas as pd
import transformers
import csv
import datetime
import snscrape.modules.twitter as sntwitter
from utils import *

st.set_page_config(layout="wide")
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

image = img.open('images/iim_indore.jpg')  #logo
st.image(image, width=350)  # logo width
st.title("IPBA- Batch 13")
st.title('Sentiment Analysis')
st.markdown("""
<-- Search a hashtag in the sidebar to run the tweets analyzer!
""")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
     border-radius: 0.3rem;
}
div.stButton > button:hover {
    background-color: #60ff10;
    color:#f00333;
    }
</style>""", unsafe_allow_html=True)


st.sidebar.header('Search through twitter hashtag!') #sidebar title


with st.form(key ='form_1'):
    with st.sidebar:
        hashtag = st.text_input('Enter hashtag', "#India", help='Ensure hashtag does not contain spaces!')
        num_of_tweets = st.number_input('Maximum number of tweets', min_value=20, max_value=50, value=20, step=5, help='Returns the most recent tweets within the last 7 days')
        submitted1 = st.form_submit_button(label='Analyse tweets ')

    method_expander = st.sidebar.expander("Methodology")
    method_expander.markdown("""
    * Applying the [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
    * Sentiments categorised under : Positive,  Neutral and Negative
    """)

# # Loading message for users
# with st.spinner('Loading IPBA 13 Sentiment analyser...'):
#     time.sleep(2)
#     # Keyword or hashtag
#     if submitted1:
#         st.success('🎈Done! You searched for the last ' + str(num_of_tweets) + ' tweets that used #' + hashtag)
#
#

# with st.form(key='Analyse_form'):
    try:
        sentence = st.text_input('Enter a sentence to analyse the sentiment')
        # no_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (maximum 50 tweets)', 0, 50, 10)
        submit_button = st.form_submit_button(label='Submit')
        st.text("Or")
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        file_list = os.listdir(datasets_dir)
        file_list.insert(0, None)
        # Create a dropdown menu of file names
        selected_file = st.selectbox("select an existing dataset: ", file_list, index=0)
        count = st.number_input('Maximum number of tweets/comments to analyse', min_value=10, max_value=2000, value=10, step=100,
                        help='Analyses the sentiment of set number of tweets/comments')
        submit_button_existing_dataset = st.form_submit_button(label='Analyse')
        if submit_button:
            with st.spinner("Getting the sentiment of your sentence ........"):

                if sentence:
                    analyse_sentence = SentimentAnalyser(sentence)
                    analyse_sentence.analyse()
                else:
                    st.write("Enter a sentence to get the sentiment...!")

        elif submit_button_existing_dataset:
            if selected_file is not None:
                with st.spinner("analysing the dataset...."):
                    analyse_dataset = SentimentAnalyser(os.path.join(os.getcwd(), "datasets", selected_file))
                    analyse_dataset.analyse_dataset(count)
            else:
                st.write("Select an existing dataset")

            # st.write(selected_file)
    except Exception as e:
        print(e)
