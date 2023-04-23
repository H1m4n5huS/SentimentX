import os
import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter
import streamlit as st
import transformers

# Define the hashtag and date range to search for
start_date = datetime.datetime.now() - datetime.timedelta(days=300)
end_date = datetime.datetime.now()

class SentimentAnalyser:
    def __init__(self, sentence):
        self.sentence = sentence
        self.model = transformers.pipeline('sentiment-analysis')

    def analyse(self):
        result = self.model(self.sentence)[0]
        label = result['label']
        if label == 'POSITIVE':
            st.write(f"The sentiment of '{self.sentence}' is positive")
        else:
            st.write(f"The sentiment of '{self.sentence}' is negative")


# Function to fetch tweets
def get_tweets(query, num_tweets):
    tweets_list = []
    query = f"{query} since:{start_date.date()} until:{end_date.date()}"
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= num_tweets:
            break
        tweets_list.append([tweet.date, tweet.content])
    return pd.DataFrame(tweets_list, columns=['date', 'tweet'])


st.set_page_config(layout="wide")
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

st.title("IPBA- Batch 13")
st.title('Sentiment Analysis')
st.markdown("""
<-- Search a hashtag in the sidebar to run the tweets analyzer!
""")

st.sidebar.header('Search through twitter hashtag!') #sidebar title

with st.form(key ='form_1'):
    with st.sidebar:
        hashtag = st.text_input('Enter hashtag', "#India", help='Ensure hashtag does not contain spaces!')
        num_of_tweets = st.number_input('Maximum number of tweets', min_value=20, max_value=50, value=20, step=5, help='Returns the most recent tweets')
        submitted1 = st.form_submit_button(label='Analyse tweets ')

try:
    if submitted1:
        st.success(f'Done! You searched for the last {num_of_tweets} tweets that used #{hashtag}')
        st.text("Here are the tweets for analysis:")
        tweets_df = get_tweets(hashtag, num_of_tweets)
        st.dataframe(tweets_df)

        with st.form(key='form_2'):
            sentence = st.text_input('Enter a sentence to analyse the sentiment')
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                analyser = SentimentAnalyser(sentence)
                analyser.analyse()

            st.text("Or")

            manual_tweet = st.text_input('Enter the tweet text')
            manual_sentiment = st.selectbox("Select the sentiment: ", ["Positive", "Negative"])
            submit_button_manual_tweet = st.form_submit_button(label='Analyse Manual Tweet')

            if submit_button_manual_tweet:
                if manual_tweet:
                    if manual_sentiment == "Positive":
                        st.write(f"The sentiment of '{manual_tweet}' is positive")
                    else:
                        st.write(f"The sentiment of '{manual_tweet}' is negative")
                else:
                    st.write("Please enter the tweet text")

        st.text("Or")

        with st.form(key='form_3'):
            datasets_dir = os.path.join(os.getcwd(), "../datasets")
            file_list = os.listdir(datasets_dir)
            file_list.insert(0, None)
            selected_file = st.selectbox("select an existing dataset: ", file_list, index=0)
            submit_button_existing_dataset = st.form_submit_button(label='
