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

comments_mcd = []
postmcd.comments.replace_more(limit=None)
for comments in postmcd.comments.list():
    comments_mcd.append(comments.body)

#print(comments_mcd, '\n')
#print('Total Comments MCD=', (len(comments_mcd)))

list1 = comments_mcd
list1 = [str(i) for i in list1]
string_uncleaned_mcd = ' , '.join(list1)

string_noemoji_mcd = emoji.get_emoji_regexp().sub(u'', string_uncleaned_mcd)
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string_mcd = tokenizer.tokenize(string_noemoji_mcd)
#print(tokenized_string)

lower_string_tokenized_mcd = [word.lower() for word in tokenized_string_mcd]
#print(lower_string_tokenized)

nlp_mcd = en_core_web_sm.load()
all_stopwords_mcd = nlp_mcd.Defaults.stop_words
text_mcd = lower_string_tokenized_mcd
tokens_without_sw=[word for word in text_mcd if not word in all_stopwords_mcd]
#print(tokens_without_sw)

lemmatizer_mcd = WordNetLemmatizer()
lemmatized_tokens_mcd = ([lemmatizer_mcd.lemmatize(w) for w in tokens_without_sw])
print(lemmatized_tokens_mcd)
cleaned_output_mcd = lemmatized_tokens_mcd

#shows that the number of words uncleaned vs cleaned. Cleaned is much less.
print("Original length of words = ", (len(string_uncleaned_mcd)))
print("Number of words for final output = ", (len(cleaned_output_mcd)))


#comments amazon
comments_amazon = []
postamazon.comments.replace_more(limit=None)
for comments in postamazon.comments.list():
    comments_amazon.append(comments.body)

