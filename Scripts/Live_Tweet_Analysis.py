# import tweepy as tw
import streamlit as st
import pandas as pd
from transformers import pipeline
import csv
import datetime
import snscrape.modules.twitter as sntwitter


# classifier = pipeline('sentiment-analysis')
st.title("IPBA- Batch 13")
st.title('Live Twitter Sentiment Analysis')
st.markdown('Get the sentiment labels of live tweets!')

# Define the hashtag and date range to search for
hashtag = "#boycottgillette"
start_date = datetime.datetime(2019, 1, 13)
end_date = datetime.datetime(2019, 1, 31)

# Create a file to store the tweets
# with open(f"{hashtag}.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["id", "content", "date"])


def run():
    with st.form(key='Enter name'):
        search_words = st.text_input('Enter the hashtag for which sentiment is to be analysed')
        no_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (maximum 50 tweets)', 0, 50, 10)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:

        # Use snscrape to search for tweets with the hashtag and within the specified date range
        tweet_list = sntwitter.TwitterSearchScraper(
                f'{hashtag} lang:en').get_items()
        print(tweet_list)
        # tweet_list = [i.text for i in tweets]
        # output = [i for i in classifier(tweet_list)]
        # labels = [output[i]['label'] for i in range(len(output))]
        # df = pd.DataFrame(list(zip(tweet_list, labels)),
        #                   columns=['Latest ' + str(no_of_tweets) + ' tweets' + ' on ' + search_words, 'Sentiment'])
        # st.write(df)


if __name__ == '__main__':
    run()
