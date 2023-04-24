import csv
import os
import datetime
import snscrape.modules.twitter as sntwitter
# import googleapiclient
# from googleapiclient.discovery import build


# start_date = datetime.datetime(2019, 1, 13)
# end_date = datetime.datetime(2019, 1, 31)


class TwitterExtract:
    def __init__(self, hashtag: str, start_date: datetime.datetime, end_date: datetime.datetime):
        """
        Class constructor for twitter comments extraction
        : param hashtag: hashtag for which comments are searched
        : param start_date: start time from which comments should be extracted
        : param end_date: end time till which comments should be extracted
        """
        self._hashtag = hashtag
        self.start_date = start_date
        self.end_date = end_date

    @property
    def hashtag(self):
        return self._hashtag

    @hashtag.setter
    def hashtag(self, value):
        if len(value)==0:
            raise Exception("Hashtag cannot be empty")
        elif '#' not in value:
            self._hashtag = '#' + value
        else:
            self._hashtag = value

    def get_tweets(self):

        datasets_dir = os.path.join(os.getcwd(), "datasets")
        filename = os.path.join(datasets_dir, self._hashtag[1:])
        # Create a file to store the tweets
        with open(f"{filename}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "content", "date"])

            # Use snscrape to search for tweets with the hashtag and within the specified date range
            for tweet in sntwitter.TwitterSearchScraper(
                    f'{self._hashtag} since:{self.start_date.date()} until:{self.end_date.date()} lang:en').get_items():
                # Only include tweets that were created within the specified time range
                tweet_date = tweet.date.replace(tzinfo=None)
                if self.start_date <= tweet_date <= self.end_date:
                    writer.writerow([tweet.id, tweet.content, tweet.date])


class YoutubeExtract:
    def __init__(self, hashtag: str, video_id: str):
        self._hashtag = hashtag
        self._video_id = video_id
        self.api_key = 'AIzaSyCjjoUrGst995BTwuDaBkOrawkMB9oxffQ'

    @property
    def hashtag(self):
        return self._hashtag

    @hashtag.setter
    def hashtag(self, value):
        if len(value) == 0:
            raise Exception("Hashtag cannot be empty")
        elif '#' not in value:
            self._hashtag = '#' + value
        else:
            self._hashtag = value

    @property
    def video_id(self):
        return self._video_id

    @video_id.setter
    def video_id(self, value):
        if len(value) == 0:
            raise Exception("video id cannot be empty")
        else:
            self._video_id = value

    # function to extract comments and replies and write to CSV
    def video_comments_to_csv(self):
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        filename = os.path.join(datasets_dir, self._hashtag[1:])
        # creating youtube resource object
        # youtube = build('youtube', 'v3', developerKey=self.api_key)

        # create a CSV file for writing the data to
        with open(f'comments_for_video_{self._video_id}.csv', 'w', newline='', encoding='utf-8') as file:
            # create a CSV writer object
            writer = csv.DictWriter(file,
                                    fieldnames=['Comment Number', 'Reply Number', 'Like Count', 'Published At', 'Text'])
            # write the header row to the CSV file
            writer.writeheader()

            # counters for comments and replies
            comment_counter = 1
            reply_counter = 1

            # retrieve the video comments
            # video_response = youtube.commentThreads().list(
            #     part='snippet',
            #     videoId=self._video_id,
            #     textFormat='plainText'
            # ).execute()

            # iterate through the video comments
        #     # while video_response:
        #         # iterate through each item in the video comments
        #         for item in video_response['items']:
        #             # extract the comment data
        #             comment = item['snippet']['topLevelComment']['snippet']
        #             comment_text = comment['textDisplay']
        #             comment_published_at = comment['publishedAt']
        #             comment_like_count = comment['likeCount']
        #             # write the comment data to the CSV file
        #             writer.writerow({'Comment Number': comment_counter,
        #                              'Reply Number': '',
        #                              'Like Count': comment_like_count,
        #                              'Published At': comment_published_at,
        #                              'Text': comment_text})
        #             # increment the comment counter
        #             comment_counter += 1
        #
        #             # retrieve the replies to the comment
        #             if item['snippet']['totalReplyCount'] > 0:
        #                 reply_response = youtube.comments().list(
        #                     part='snippet',
        #                     parentId=item['snippet']['topLevelComment']['id'],
        #                     textFormat='plainText'
        #                 ).execute()
        #
        #                 # iterate through the replies to the comment
        #                 for reply in reply_response['items']:
        #                     reply_text = reply['snippet']['textDisplay']
        #                     reply_published_at = reply['snippet']['publishedAt']
        #                     if 'likeCount' in reply:
        #                         reply_like_count = reply['likeCount']
        #                     else:
        #                         reply_like_count = 0
        #                     # write the reply data to the CSV file
        #                     writer.writerow({'Comment Number': '',
        #                                      'Reply Number': reply_counter,
        #                                      'Like Count': reply_like_count,
        #                                      'Published At': reply_published_at,
        #                                      'Text': reply_text})
        #                     # increment the reply counter
        #                     reply_counter += 1
        #
        #         # check if there are more comments to retrieve
        #         if 'nextPageToken' in video_response:
        #             video_response = youtube.commentThreads().list(
        #                 part='snippet',
        #                 videoId=self._video_id,
        #                 textFormat='plainText',
        #                 pageToken=video_response['nextPageToken']
        #             ).execute()
        #         else:
        #             break
        # print(f"CSV file saved at {filename}")