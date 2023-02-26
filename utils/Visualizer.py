import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display, Image


class PostProcessor:
    def __init__(self, data):
        self._sentiments = data

    def plot_sentiments(self):
        """
        Plot the sentiments of the tweets using a histogram.
        """
        plt.hist(self._sentiments, bins=20)
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        plt.title('Sentiment Analysis')
        plt.show()

    @property
    def data(self):
        return self._sentiments

    @data.setter
    def data(self, value):
        if not value["polarity"]:
            raise Exception("No Polarity column in the provided data")
        elif len(value["polarity"]) != 0:
            self._sentiments = value

    def plot_word_cloud(self) -> None:
        """
        Function that displays a word cloud
        :return: None
        """
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10).generate(self._sentiments["polarity"])
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        # Will work only in jupyter notebook.
        display(Image(filename='twitter_wordcloud.png'))
