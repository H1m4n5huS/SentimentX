#!/usr/bin/env python
# coding: utf-8

# In[33]:


#importing required libraries
import pandas as pd
import numpy as np 
import nltk
import re
import urlextract
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns


#  Preprocesssing textual data


from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
pd.set_option('display.max_colwidth', 1000)

# additional nltk resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer


# Check Performance
from sklearn.metrics import classification_report


# In[4]:


# Importing required data AND Overview
df = pd.read_csv(r"C:\Users\sukan\ipba\csv\comments_for_video_wGBoppe3ZqQ.csv")
df.head(25)


# In[5]:


df.shape[0]


# In[6]:


df.isna().sum()


# In[7]:


df.columns


# In[8]:


# Changing name of 'text' column to 'Comments' column
df.rename(columns = {'Text':'Comments'}, inplace = True)


# In[9]:


df.columns


# In[10]:


df.isna().sum()


# In[11]:


##  Descriptive Statistics


# In[12]:


## Creating Length Feature
df['text_length'] = df['Comments'].apply(lambda x : len(x))


# In[16]:


df['text_length'].plot.hist(bins = 5)


# In[17]:


# Null or missing value rows detection

null_rows = df[df['Comments'].isnull()]

print(null_rows)


# In[19]:


## converting Comments to strings
df['Comments'] = df['Comments'].astype(str)
df.shape


# In[20]:


#### Cleaning--- puntuations removal,lowercase, extra spaces & url removals


# In[21]:


##Removing punctuation

import string
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# In[22]:


df['Comments'].apply(remove_punctuations)


# In[23]:


## lowercase
df['Comments'].apply(lambda x: x.lower())


# In[24]:


# Commit to Table
df['Comments'] = df['Comments'].apply(lambda x: x.lower())


# In[25]:


## Extra spaces and URLs removal
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'https?\S+', '', x).strip())


# In[26]:


df.head(20)


# In[27]:


##### Tokenization using tweet tokenizer


# In[28]:


import nltk
from nltk.tokenize import TweetTokenizer

# Instantiate the TweetTokenizer
tokenizer = TweetTokenizer()

# Define a function to tokenize a tweet
def tokenize_tweet(tweet):
    # Use the TweetTokenizer to tokenize the tweet
    tokens = tokenizer.tokenize(tweet)
    return tokens

# Apply the tokenization function to the 'Comments' column
df['Tokens'] = df['Comments'].apply(tokenize_tweet)


# In[29]:


df.head(20)


# In[30]:


### removing stopwords,special characters and numbers


# In[31]:


# Retrieve Stopwords
stop = stopwords.words('english')


# In[32]:


# Tokenize and Remove Stop Words
df['Comments'].apply(lambda x: [word for word in x.split() if word not in stop])


# In[36]:


df['Comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[37]:


# Join the words back
df['Comments'] = df['Comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[38]:


df.head(10)


# In[40]:


##Removing numbers

df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'\d+', '', x))


# In[44]:


##Removing non-ASCII improperly encoded characters

def remove_invalid_chars(text):
    # encode the text string using ASCII encoding
    encoded_text = text.encode('ascii', 'ignore')
    # decode the encoded text using ASCII encoding
    decoded_text = encoded_text.decode('ascii')
    return decoded_text
# apply the remove_invalid_chars function to the 'Comments' column
df['Comments'] = df['Comments'].apply(remove_invalid_chars)


# In[45]:


df.head(15)


# In[46]:


### Lemmatize


# In[47]:


from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


# In[48]:


df['Comments'].apply(lambda x: lemmatizer.lemmatize(x))


# In[49]:


# Commit to Table

df['Comments'] = df['Comments'].apply(lambda x: lemmatizer.lemmatize(x))


# In[50]:


### POS Tagging


# In[51]:


def pos_tagging(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    # Perform POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    # Return the tagged tokens
    return tagged_tokens


# In[52]:


# Applying the POS tagging function to the 'Comments' column
df['POS_Tagged'] = df['Comments'].apply(pos_tagging)


# In[53]:


df['POS_Tagged']


# In[54]:


### Model selections and preparations
# Huggingface pipeline for quick and easy way to run sentiment predictions
# VADER (Valence Aware Dictionary and sEntiment Reasoner) - Bag of words approach
# Roberta Pretrained Model


# In[55]:


## 1-[Huggingface pipeline]--- quick and easy way to run sentiment predictions
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[56]:


df.head(15)


# In[57]:


sent_pipeline('probably children trafficked entertainment industry	')


# In[61]:


sent_pipeline('madeleine mccann used carry teddy bear around time	')


# In[59]:


sent_pipeline('thank this.	')


# In[62]:


## 2.[VADER Seniment Scoring]---We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.

#This uses a "bag of words" approach:1.Stop words are removed 2.each word is scored and combined to a total score.


# In[63]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[64]:


sia = SentimentIntensityAnalyzer()


# In[65]:


sia.polarity_scores('probably children trafficked entertainment industry	')


# In[66]:


sia.polarity_scores('madeleine mccann used carry teddy bear around time	')


# In[67]:


sia.polarity_scores('thank this.	')


# In[79]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Comments']
    myid = row['Comment Number']
    res[myid] = sia.polarity_scores(text)


# In[80]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})


# In[82]:


# Now we have sentiment score and metadata
vaders.head(25)


# In[ ]:


## 3.[Roberta Pretrained Model]--- 1.Use a model trained of a large corpus of data. , 2.Transformer model accounts for the words but also the context related to other words. 3 compare result with vader


# In[86]:


## Importing model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[87]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[93]:


## To check and compare polarity scores from vader against roberta we'll create an object( lets say Sample) containing random tweet.
Sample = df['Comments'][8]
print(Sample)


# In[94]:


sia.polarity_scores(Sample)


# In[96]:


# Run for Roberta Model
encoded_text = tokenizer(Sample, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[97]:


def polarity_scores_roberta(Sample):
    encoded_text = tokenizer(Sample, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[98]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Comments']
        myid = row['Comment Number']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[101]:


## Comparing the results of the two models- Vader vs Roberta
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df.columns


# In[102]:


results_df.head(25)


# In[ ]:





# In[ ]:





# In[ ]:




