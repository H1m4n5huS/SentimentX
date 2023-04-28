import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from IPython.display import display, Image


class PostProcessor:
    def __init__(self, data):
        self._sentiments = data
        self.col1, self.col2, self.col3 = st.columns(3)

    def plot_sentiments(self, polarity):
        # """
        # Plot the sentiments of the tweets using a histogram.
        # """
        # plt.hist(self._sentiments, bins=5)
        # plt.xlabel('Sentiment')
        # plt.ylabel('Frequency')
        # plt.title(f'Sentiment Analysis {polarity}')
        # st.pyplot()
        pass

    def plot_pie_chart(self, df):

        with self.col1:
            counts = df.value_counts()

            # Create data for the pie chart
            labels = counts.index
            sizes = counts.values

            # Create a pie chart
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%0.01f%%', startangle=90)
            ax1.axis('equal')

            # Display the chart in Streamlit
            st.pyplot(fig1)
    @property
    def data(self):
        return self._sentiments

    @data.setter
    def data(self, value):
        """
         Setter function which checks for the polarity column
         """
        if not value["score"]:
            raise Exception("No score column in the provided data")
        elif len(value["score"]) != 0:
            self._sentiments = value

    def plot_word_cloud(self, word_count) -> None:

        with self.col2:
            # generate the wordcloud
            all_text = ' '.join(self._sentiments["Comment"])
            wordcloud = WordCloud(width=2000, height=1800, background_color='white').generate(all_text)

            # plot the wordcloud
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()
            st.text(f"Total words analysed : {word_count}")




