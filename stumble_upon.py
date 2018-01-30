import pandas as pd
from textblob import TextBlob
import json
import nltk
import string
import math
from nltk.corpus import stopwords
from collections import Counter
from nltk import ngrams

# constant
ignorechars = [',', '.','-','--', '&', ';', ':', '?','#','>','<','=','[',']',')','(', '|','com','html', '!']
word_list=['cooking', 'baking', 'top', 'food', 'look', 'day', '10', 'eat', 'try', 'serve', 'cool', 'little', 're',
          'heat', '12', 'butter', 'video', 'read', 'help', 'call', 'week', 'love', 'add', 'sugar', 'minutes',
          'start', 'run', 'love', 'eggs', 'flavor','cheese', 'own', '2011', '20', 'quick','brown', 'home',
          'chocolate', 'mix', 'people', 'tablespoons', 'blog', 'completely', 'water', 'including', 'ingredients', 
          'prepared', 'preheat', 'natural', 'friends', 'create', 'remove', 'inspired', 'recipes', 'ice', 'remove',
          'test', 'salt', 'line', 'write', 'news', 'cup', 'dish', 'food', 'fashion', 'article', 'image', 'milk',
          'light', 'store', 'slice','person','family','season','hand','create','pan', 'fun', 'hit', 'author', 
           'teaspoon', 'months', 'heavy', 'game', 'kitchen', 'recently', 'alternative', 'paper','slice','bowl',
          'cream', 'combine', 'vegetable', 'follow', 'roll', 'filling']


def get_tokens(rev):
	lowers = str(rev).lower() 
	#remove the punctuation using the character deletion step of translate
	no_punctuation = lowers.translate(string.punctuation)
	tokens = nltk.word_tokenize(no_punctuation)
	return (tokens)

def json_to_list(json_col):
	boiler = []
	for text in df[json_col]:
		jsn = json.loads(text)
		boiler.append(jsn)
	return(boiler)

def clean_text(text):
	filtered_text=[]
	for t in text:
		if t is not None:
			tokens = get_tokens(t)
			filtered = [w for w in tokens if not w in stopwords.words('english')]
			filtered = [w for w in filtered if not w in ignorechars]
			filtered_text.append(filtered)
		else:
			filtered_text.append(None)
	return (filtered_text)

def list_to_string(l):
	pool = []

	for text in l:
		if text is not None:
			words = ' '.join(text)
			pool.append(words)
		else:
			pool.append(None)
	return (pool)

def get_ngrams(sentence, n):
	gramlist=[]
	gram = []
	for text in sentence:
		if text is not None:
			three_grams=ngrams(text.split(), n)
			grams= [' '.join(gram) for gram in three_grams]
			gramlist.append(grams)
		else:
			gramlist.append(None)
	return (gramlist)
                
def append_column(lst):
	se = pd.Series(lst)
	return (se.values)

def tf(word, blob):
    if len(blob.words)==0:
        return 0
    else:
        return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def set_tfidf(word, bloblist):
    scores=[]
    for i, blob in enumerate(bloblist):
        scores.append(tfidf(word, blob, bloblist))
    se_word = pd.Series(scores)
    df_notNull[word]=se_word.values

# -- -- - -- --- - - -- MAIN PROGRAM -- -- - -- --- - - -- #


df = pd.read_csv("test-stumble_upon.tsv", sep="\t")
print('read success')

boiler = json_to_list('boilerplate')
print('checking sample content: ')
print(boiler[0], '\n')

# transform body, title, url in boilerplate as columns
body=[]
title=[]
url=[]

for item in boiler:
    body.append(item.get('body'))
    title.append(item.get('title'))
    url.append(item.get('url'))

print('body, title, and url succesfully created\n')

print('start cleaning text')
# filter from stopwords & unwanted chars
filtered_title = clean_text(title)
print('title cleaned')
filtered_body = clean_text(body)
print('body cleaned')
filtered_url = clean_text(url)
print('url cleaned')
print('start appending list item into sentences')
# append list item in each row to be sentences
title = list_to_string(filtered_title)

body = list_to_string(filtered_body)

url = list_to_string(filtered_url)

print('start creating n-grams')
title_grams=get_ngrams(title, 2)

body_grams=get_ngrams(body, 2)

url_grams=get_ngrams(url, 2)
print('appending new columns')
df['title_3grams']= append_column(title_grams)
df['body_3grams']= append_column(body_grams)
df['url_3grams']= append_column(url_grams)
print('data frame operations succeed')
print(df.head(n=5))
save = input('save df? (Y/N)')
if save == 'Y':
	df.to_csv('train_ngrams.tsv', sep='\t')


