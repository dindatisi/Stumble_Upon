import pandas as pd
from textblob import TextBlob
import json
import nltk
import string
from nltk.corpus import stopwords
from collections import Counter

# constant
ignorechars = [',', '.','-','--', '&', ';', ':', '?','#','>','<','=','[',']',')','(', '|','com','html', '!']

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

	for text in filtered_url:
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
                
def append_column(column_name, lst):
	se = pd.Series(lst)
	return (se.values)


# -- -- - -- --- - - -- MAIN PROGRAM -- -- - -- --- - - -- #


df = pd.read_csv("train-stumble_upon.tsv", sep="\t")
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
df['body_3grams']= append_column(bodygrams)
df['url_3grams']= append_column(url_grams)
print('data frame operations succeed')
print(df.head(n=5))
save = input('save df? (Y/N)')
if save == 'Y':
	df.to_csv('train_ngrams.tsv', sep='\t')


