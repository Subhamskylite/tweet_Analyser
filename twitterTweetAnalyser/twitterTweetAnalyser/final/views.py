from django.shortcuts import render
import twint
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import re
from nltk.corpus import stopwords
nltk.download('punkt')
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

def stats(request):
    print("stats")
    return render(request, 'stats.html',{'title':'Model Evalution','acti':'nav-acti'})

def clean_tweet(tweet):
    # Remove URLs
    
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"https?://\S+|www\.\S+", "", tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # Remove usernames
    tweet = re.sub(r"@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub(r".@", "", tweet)
    tweet = re.sub("[^0-9 ]+", "", tweet)
    # Remove hashtags
    tweet = re.sub(r"#", "", tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Tokenize
    tokens = nltk.word_tokenize(tweet)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    tweet = " ".join(filtered_tokens)
    # Handle negation
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    tweet = " ".join(stemmed_tokens)
    tweet = re.sub(r"\b(?:not|never|no)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\S+)', r'\1NOT_\2', match.group(0)), tweet)
    return tweet


# def get_sentiment(text):
#     analyzer = SentimentIntensityAnalyzer()
#     result = analyzer.polarity_scores(text)
#     compound = float(result['compound'])  # Convert compound to float
#     if compound > 0.05:
#         sentiment = 'positive'
#     elif compound < -0.05:
#         sentiment = 'negative'
#     else:
#         sentiment = 'neutral'
#     return sentiment

# # Load the 'tweet' DataFrame from your code
# # Replace this with your actual code to load the 'tweet' DataFrame

# # Convert 'polarity' column to float
#     tweet['polarity'] = tweet['polarity'].astype(float)

# # Apply get_sentiment() function to 'text' column to get 'sentiment' column
#     tweet['sentiment'] = tweet['text'].apply(get_sentiment)

# # Use np.where() to assign sentiment labels based on 'polarity' values
#     tweet['sentiment'] = np.where(tweet['polarity'] > 0.05, 'positive',
#                               np.where(tweet['polarity'] < -0.05, 'negative', 'neutral'))


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    result = analyzer.polarity_scores(text)
    compound = float(result['compound'])  # Convert compound to float
    if compound > 0.05:
        sentiment = 'positive'
    elif compound < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    subjectivity = float(result['compound'])  # Use compound as subjectivity for example
    # Convert 'polarity' column to float
    
    return sentiment

def  predict(request):
    print("predict")
    outputRes= ''
    tweet=''
    val=0
    total_neutral_tweets=0
    total_positive_tweets=0
    total_negative_tweets=0
    keyword=''
    totalTweets=0
    img_base64=""
    if request.method == 'POST':
        outputRes='done'
        c = twint.Config()
        c.Search = request.POST['Search']
        keyword=request.POST['Search'].upper()
        c.language = "en"
        c.Store_object = True
        c.Pandas = True

        c.Pandas_clean = True
        c.Store_csv = False
        c.Limit=200
        c.Columns = ["username", "tweet"]
        # outputRes = request.POST['keyword']
        twint.run.Search(c)
        tweet= twint.storage.panda.Tweets_df[["username", "tweet", "language","date"]]

        # Clean tweet text
        tweet['clean_tweet'] = tweet['tweet'].apply(clean_tweet)
        tweet['sentiment'] = tweet['tweet'].apply(get_sentiment)
        # print(tweet['sentiment'])
        # mask = tweet['clean_tweet'].notnull()

# Use the .loc indexer to assign values to 'polarity' and 'subjectivity' columns
        # tweet.loc[mask, ['polarity', 'subjectivity']] = tweet.loc[mask, 'clean_tweet'].apply(get_sentiment).apply(lambda x: x[0])
       
        # tweet['polarity'] = tweet['polarity'].map({'positive': 1, 'neutral': 0, 'negative': -1}).astype(float)
        
# Assign values to 'sentiment' column based on 'polarity' values
        # tweet['sentiment'] = np.where(tweet['polarity'] >  0.05, 'positive', np.where(tweet['polarity'] < -0.05, 'negative', 'neutral'))
         # Count total positive, negative, and neutral tweets
        
        total_positive_tweets = tweet[tweet['sentiment'] == 'positive'].shape[0]
        total_negative_tweets = tweet[tweet['sentiment'] == 'negative'].shape[0]
        total_neutral_tweets = tweet[tweet['sentiment'] == 'neutral'].shape[0]
        val=1
        totalTweets=total_positive_tweets+total_negative_tweets+total_neutral_tweets
        
        total_neutral_tweets=round((total_neutral_tweets/totalTweets)*100,2)
        total_positive_tweets=round((total_positive_tweets/totalTweets)*100,2)
        total_negative_tweets=round((total_negative_tweets/totalTweets)*100,2)

        print("Total Positive Tweets:", total_positive_tweets)
        print("Total Negative Tweets:", total_negative_tweets)
        print("Total Neutral Tweets:", total_neutral_tweets)
        # Create a string of all the tweets with positive sentiment
        # positive_tweets = ' '.join(tweet[tweet['sentiment'] == 'positive']['tweet'])
        
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        tweet = ' '.join('clean_tweet')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweet)
    
    # Generate the wordcloud as an image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
    
    # Convert the image to base64 string
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = buf.getvalue()
        buf.close()
    
    # Pass the image data to the template



    return render(request,'predict.html', {'image': img_base64,'value':val,'searchWord':keyword,'total':totalTweets,'neutralTweets':total_neutral_tweets,'positiveTweets':total_positive_tweets,'negativeTweets':total_negative_tweets,'log':tweet, 'title':'Try to Predict !','acti':'nav-acti'})
    