import praw
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from praw.models import MoreComments
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
import emoji
import re
import en_core_web_sm
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import string
import sys
import markovify
import random


def main():
    """Scraping data from two reddit threads (McDonalds and Target employee reviews, then doing text analysis, sentiment analysis, and Markov sentence generation"""
    #scraping reddit for comments in both threads
    reddit = praw.Reddit(client_id="n6o_85mCpVP-PSLDLpSMHA", client_secret="evKdE3d58KYGe16q_OjeB9lGLiJ_9g", user_agent="APImcd")

    postmcd = reddit.submission(id='1wddtd')
    posttarget = reddit.submission(id='p38bg9')

    #Data cleaning
    #puts comments into a list
    comments_mcd = []
    postmcd.comments.replace_more(limit=None)
    for comments in postmcd.comments.list():
        comments_mcd.append(comments.body)


    list1 = comments_mcd
    list1 = [str(i) for i in list1]
    string_uncleaned_mcd = ' , '.join(list1)

    #removes emojis from comments
    string_noemoji_mcd = emoji.get_emoji_regexp().sub(u'', string_uncleaned_mcd)
    tokenizer_mcd = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
    tokenized_string_mcd = tokenizer_mcd.tokenize(string_noemoji_mcd)
    #print(tokenized_string)

    #Removes capital letters to make words more consistent with each other
    lower_string_tokenized_mcd = [word.lower() for word in tokenized_string_mcd]
    #print(lower_string_tokenized)

    #Removes stop words
    nlp_mcd = en_core_web_sm.load()
    all_stopwords_mcd = nlp_mcd.Defaults.stop_words
    text_mcd = lower_string_tokenized_mcd
    tokens_without_sw_mcd=[word for word in text_mcd if not word in all_stopwords_mcd]
    #print(tokens_without_sw)

    #Lemmatizes words. Basically words that are similar can be grouped together in same category.
    lemmatizer_mcd = WordNetLemmatizer()
    lemmatized_tokens_mcd = ([lemmatizer_mcd.lemmatize(w) for w in tokens_without_sw_mcd])
    print(lemmatized_tokens_mcd)
    cleaned_output_mcd = lemmatized_tokens_mcd

    #prints that the number of words uncleaned vs cleaned. Cleaned is much less.
    print("Original length of words = ", (len(string_uncleaned_mcd)))
    print("Number of words for final output = ", (len(cleaned_output_mcd)))


    #Scraping comments from Target reddit thread
    comments_target = []
    posttarget.comments.replace_more(limit=None)
    for comments in posttarget.comments.list():
        comments_target.append(comments.body)

    list2 = comments_target
    list2 = [str(i) for i in list2]
    string_uncleaned_target = ' , '.join(list2)

    #Removes emoji from comments
    string_noemoji_target = emoji.get_emoji_regexp().sub(u'', string_uncleaned_target)
    tokenizer_target = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
    tokenized_string_target = tokenizer_target.tokenize(string_noemoji_target)
    #print(tokenized_string)

    #Removes capital letters from words
    lower_string_tokenized_target = [word.lower() for word in tokenized_string_target]
    #print(lower_string_tokenized)

    #Removes stop words from comments
    nlp_target = en_core_web_sm.load()
    all_stopwords_target = nlp_target.Defaults.stop_words
    text_target = lower_string_tokenized_target
    tokens_without_sw_target=[word for word in text_target if not word in all_stopwords_target]
    #print(tokens_without_sw)

    #Lemmatizes words or basically groups certain similar words into the same category
    lemmatizer_target = WordNetLemmatizer()
    lemmatized_tokens_target = ([lemmatizer_target.lemmatize(w) for w in tokens_without_sw_target])
    print(lemmatized_tokens_target)
    cleaned_output_target = lemmatized_tokens_target

    #shows that the number of words uncleaned vs cleaned. Cleaned is much less.
    print("Original length of words = ", (len(string_uncleaned_target)))
    print("Number of words for final output = ", (len(cleaned_output_target)))

    #shows that the cleaned output is a list
    print(type(cleaned_output_target))
    print(type(cleaned_output_mcd))

    #VADER Sentiment Analysis for MCD
    sia_mcd = SIA()
    results_mcd = []

    for sentences_mcd in cleaned_output_mcd:
        pol_score_mcd = sia_mcd.polarity_scores(sentences_mcd)
        pol_score_mcd['words'] = sentences_mcd
        results_mcd.append(pol_score_mcd)
    pd.set_option('display.max_columns', None, 'max_colwidth', None)
    df_mcd = pd.DataFrame.from_records(results_mcd)
    print(df_mcd)

    df_mcd['label'] = 0
    df_mcd.loc[df_mcd['compound'] > 0.05, 'label'] = 1
    df_mcd.loc[df_mcd['compound'] < -0.05, 'label'] = -1
    df_mcd.head()

    #output surprisingly shows there is more positive sentiment than negative sentiment for McDonalds reddit thread
    print(df_mcd.label.value_counts())

    fig, ax = plt.subplots()
    counts_mcd = df_mcd.label.value_counts(normalize = True) 
    sns.barplot(x = counts_mcd.index, y=counts_mcd, ax=ax)
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Proportion")
    plt.title('Sentiment analysis for working at McDonalds')
    plt.show()



    #VADER Sentiment Analysis for target
    sia_target = SIA()
    results_target = []

    for sentences_target in cleaned_output_target:
        pol_score_target = sia_target.polarity_scores(sentences_target)
        pol_score_target['words'] = sentences_target
        results_target.append(pol_score_target)
    pd.set_option('display.max_columns', None, 'max_colwidth', None)
    df_target = pd.DataFrame.from_records(results_target)
    print(df_target)

    df_target['label'] = 0
    df_target.loc[df_target['compound'] > 0.05, 'label'] = 1
    df_target.loc[df_target['compound'] < -0.05, 'label'] = -1
    df_target.head()

    #output shows more positive sentiment than negative sentiment for target too
    print(df_target.label.value_counts())
    fig, ax = plt.subplots()
    counts_target = df_target.label.value_counts(normalize = True) 
    sns.barplot(x = counts_target.index, y=counts_target, ax=ax)
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Proportion")
    plt.title('Sentiment analysis for working at Target')
    plt.show()


    #characterize by positive and negative word frequencies for McDonalds
    freq_mcd = {} 
    for item in cleaned_output_mcd: 
        if (item in freq_mcd): 
            freq_mcd[item] += 1
        else: 
            freq_mcd[item] = 1
    print(freq_mcd)

    #return number of different words for McDonalds reddit thread excluding stop words
    print(f"There are {len(freq_mcd)} different words excluding stop words for McDonalds reddit thread")

    #sort words by frequency for mcdonalds
    sorted_freq_mcd = sorted(freq_mcd.items(), key=lambda x: x[1], reverse=True)
    print(sorted_freq_mcd)

    #sort negative words by frequency for mcdonalds
    negative_words_mcd = list(df_mcd.loc[df_mcd['label'] == -1].words)
    print(negative_words_mcd)
    negative_frequency_mcd = FreqDist(negative_words_mcd)
    #gets the 20 most common negative words for McDonalds reddit thread
    neg_freq_mcd = negative_frequency_mcd.most_common(20)
    print(neg_freq_mcd)

    #plot a bar graph to show the 20 most negative words for McDonalds by descending frequency
    #source: https://stackoverflow.com/questions/34029865/how-to-plot-bar-chart-for-a-list-in-python
    labels, ys = zip(*neg_freq_mcd)
    xs = np.arange(len(labels)) 
    width = 1
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.title("20 Most common negative words used for McDonalds reddit thread")
    plt.ylabel("Frequency")
    plt.xlabel("Negative Words")
    plt.show()


    #sort positive words by frequency for mcdonalds
    positive_words_mcd = list(df_mcd.loc[df_mcd['label'] == 1].words)
    print(positive_words_mcd)
    positive_frequency_mcd = FreqDist(positive_words_mcd)
    #gets the 20 most common positive words for McDonalds reddit thread
    pos_freq_mcd = positive_frequency_mcd.most_common(20)
    print(pos_freq_mcd)

    #plot a bar graph to show the 20 most positive words for McDonalds by descending frequency
    #source: https://stackoverflow.com/questions/34029865/how-to-plot-bar-chart-for-a-list-in-python

    labels, ys = zip(*pos_freq_mcd)
    xs = np.arange(len(labels)) 
    width = 1
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.title("20 Most common positive words used for McDonalds reddit thread")
    plt.ylabel("Frequency")
    plt.xlabel("Positive Words")
    plt.show()



    #characterize by positive and negative word frequencies for Target
    freq_target = {} 
    for item in cleaned_output_target: 
        if (item in freq_target): 
            freq_target[item] += 1
        else: 
            freq_target[item] = 1
    print(freq_target)

    #return number of different words for Target reddit thread excluding stop words
    print(f"There are {len(freq_target)} different words excluding stop words for Target reddit thread")

    #sort words by frequency for Target
    sorted_freq_target = sorted(freq_target.items(), key=lambda x: x[1], reverse=True)
    print(sorted_freq_target)

    #sort negative words by frequency for Target
    negative_words_target = list(df_target.loc[df_target['label'] == -1].words)
    print(negative_words_target)
    negative_frequency_target = FreqDist(negative_words_target)
    #gets the 20 most common negative words for Target reddit thread
    neg_freq_target = negative_frequency_target.most_common(20)
    print(neg_freq_target)

    #plot a bar graph to show the 20 most negative words for Target by descending frequency
    #source: https://stackoverflow.com/questions/34029865/how-to-plot-bar-chart-for-a-list-in-python
    labels, ys = zip(*neg_freq_target)
    xs = np.arange(len(labels)) 
    width = 1
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.title("20 Most common negative words used for Target reddit thread")
    plt.ylabel("Frequency")
    plt.xlabel("Negative Words")
    plt.show()


    #sort positive words by frequency for Target
    positive_words_target = list(df_target.loc[df_target['label'] == 1].words)
    print(positive_words_target)
    positive_frequency_target = FreqDist(positive_words_target)
    #gets the 20 most common positive words for Target reddit thread
    pos_freq_target = positive_frequency_target.most_common(20)
    print(pos_freq_target)

    #plot a bar graph to show the 20 most positive words for Target by descending frequency
    #source: https://stackoverflow.com/questions/34029865/how-to-plot-bar-chart-for-a-list-in-python

    labels, ys = zip(*pos_freq_target)
    xs = np.arange(len(labels)) 
    width = 1
    plt.bar(xs, ys, width, align='center')
    plt.xticks(xs, labels)
    plt.title("20 Most common positive words used for Target reddit thread")
    plt.ylabel("Frequency")
    plt.xlabel("Positive Words")
    plt.show()

    #print values for all keys that appear in freq_mcd that does not appear in freq_target
    res = {}
    for key in freq_mcd:
        if key not in freq_target:
            res[key] = None
        print(res)

    #print values for all keys that appear in freq_target that does not appear in freq_mcd
    res = {}
    for key in freq_target:
        if key not in freq_mcd:
            res[key] = None
        print(res)

    #print a list for all words that appear in both cleaned_output_mcd and cleaned_output_target
    #source: https://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists
    for word in cleaned_output_mcd:
        if word in cleaned_output_target:
            print(list(word))


    #Markov text generation for McDonalds 
    data_model_mcd = markovify.Text(comments_mcd)
    print("Below is the Markov text generation for McDonalds")
    for i in range(3):
        print(data_model_mcd.make_sentence())
    print("\n")


    #Markov text generation for Target
    data_model_target = markovify.Text(comments_target)
    print("Below is the Markov text generation for Target")
    for i in range(3):
        print(data_model_target.make_sentence())


if __name__ == "__main__":
    main()






