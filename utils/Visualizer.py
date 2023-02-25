import seaborn as sns
import matplotlib.pyplot as plt


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

