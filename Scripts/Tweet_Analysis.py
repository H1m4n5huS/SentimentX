# import tweepy as tw
import os
import time
import sys
import streamlit as st
from PIL import Image
import pandas as pd
import transformers
from transformers import pipeline
import csv
import datetime
import snscrape.modules.twitter as sntwitter

st.set_page_config(layout="wide")
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

image = Image.open('../iim_indore.jpg') #logo
st.image(image, width=350)  # logo width

# classifier = pipeline('sentiment-analysis')
st.title("IPBA- Batch 13")
st.title('Live Twitter Sentiment Analysis')
st.markdown("""
<-- Search a hashtag in the sidebar to run the tweets analyzer!
""")
st.markdown('Get the sentiment labels of live tweets!')

# Define the hashtag and date range to search for
start_date = datetime.datetime(2019, 1, 13)
end_date = datetime.datetime(2019, 1, 31)

st.sidebar.header('Choose Search Inputs') #sidebar title


with st.form(key ='form_1'):
    with st.sidebar:
        hashtag = st.text_input('Enter hashtag', "Tata", help='Ensure hashtag does not contain spaces!')
        num_of_tweets = st.number_input('Maximum number of tweets', min_value=20, max_value=50, value=20, step=5, help='Returns the most recent tweets within the last 7 days')
        # st.sidebar.text("") # for spacing
        submitted1 = st.form_submit_button(label='Analyse tweets ')



# Loading message for users
with st.spinner('Loading application...'):
    time.sleep(1)
    # Keyword or hashtag
    if submitted1:
        st.success('🎈Done! You searched for the last ' + str(num_of_tweets) + ' tweets that used #' + hashtag)














with st.form(key='Hashtag Entry'):
    try:
        hashtag = st.text_input('Enter the hashtag for which sentiment is to be analysed')
        no_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (maximum 50 tweets)', 0, 50, 10)
        submit_button = st.form_submit_button(label='Submit')
        st.text("Or")
        datasets_dir = os.path.join(os.getcwd(), "../datasets")
        file_list = os.listdir(datasets_dir)
        file_list.insert(0, None)
        # Create a dropdown menu of file names
        selected_file = st.selectbox("select an existing dataset: ", file_list, index=0)
        submit_button_existing_dataset = st.form_submit_button(label='Analyse')
        if submit_button:
            count = 0
            df = pd.DataFrame(columns=["id", "Comment"])
            # Use snscrape to search for tweets with the hashtag and within the specified date range
            for tweet in sntwitter.TwitterSearchScraper(f'{"mahindra"} since:{start_date.date()} until:{end_date.date()} lang:en').get_items():
                if count<=no_of_tweets:
                    print(tweet.content)
                # print(df)
        elif submit_button_existing_dataset:
            if selected_file is not None:

                st.write("analysing the dataset....")
            else:
                st.write("Select an existing dataset")
            # st.write(selected_file)
    except Exception as e:
        print(e)
