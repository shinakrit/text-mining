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

reddit = praw.Reddit(client_id="n6o_85mCpVP-PSLDLpSMHA", client_secret="evKdE3d58KYGe16q_OjeB9lGLiJ_9g", user_agent="APImcd")

postmcd = reddit.submission(id='1wddtd')
postamazon = reddit.submission(id='fyk1l8')


#for comment in postmcd.comments.list():
 #   print(comment.body)

comments_mcd = []
postmcd.comments.replace_more(limit=None)
for comments in postmcd.comments.list():
    comments_mcd.append(comments.body)

#print(comments_mcd, '\n')
#print('Total Comments MCD=', (len(comments_mcd)))

list1 = comments_mcd
list1 = [str(i) for i in list1]
string_uncleaned = ' , '.join(list1)

string_noemoji = emoji.get_emoji_regexp().sub(u'', string_uncleaned)
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(string_noemoji)
#print(tokenized_string)

lower_string_tokenized = [word.lower() for word in tokenized_string]
#print(lower_string_tokenized)

nlp = en_core_web_sm.load()
all_stopwords = nlp.Defaults.stop_words
text = lower_string_tokenized
tokens_without_sw=[word for word in text if not word in all_stopwords]
#print(tokens_without_sw)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])
print(lemmatized_tokens)
cleaned_output = lemmatized_tokens

#shows that the number of words uncleaned vs cleaned. Cleaned is much less.
print("Original length of words = ", (len(string_uncleaned)))
print("Number of words for final output = ", (len(cleaned_output)))


#comments amazon
comments_amazon = []
postamazon.comments.replace_more(limit=None)
for comments in postamazon.comments.list():
    comments_amazon.append(comments.body)

list2 = comments_amazon
list2 = [str(i) for i in list2]
string_uncleaned = ' , '.join(list2)

string_noemoji = emoji.get_emoji_regexp().sub(u'', string_uncleaned)
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(string_noemoji)
#print(tokenized_string)

lower_string_tokenized = [word.lower() for word in tokenized_string]
#print(lower_string_tokenized)

nlp = en_core_web_sm.load()
all_stopwords = nlp.Defaults.stop_words
text = lower_string_tokenized
tokens_without_sw=[word for word in text if not word in all_stopwords]
#print(tokens_without_sw)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])
print(lemmatized_tokens)
cleaned_output = lemmatized_tokens

#shows that the number of words uncleaned vs cleaned. Cleaned is much less.
print("Original length of words = ", (len(string_uncleaned)))
print("Number of words for final output = ", (len(cleaned_output)))

#print(comments_amazon, '\n')
#print('Total Comments Amazon=', (len(comments_amazon)))