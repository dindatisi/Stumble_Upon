import pandas as pd
from textblob import TextBlob
import json
import nltk
import string
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# constant
ignorechars = [',', '.','-','--', '&', ';', ':', '?','#','>','<','=','[',']',')','(', '|','com','html', '!']

# functions
def get_stemmed(words):
	print('start stemming')
	ps = PorterStemmer()
	stemmed = []
	for word in words:
		if word is not None:
			tokens = nltk.word_tokenize(str(word))
			stems=[(ps.stem(t)) for t in tokens]
			stemmed.append(stems)
		else:
			stemmed.append(None)	
	print('stemming done')
	return stemmed

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

def get_json_params(boiler):
	boilerplate=[]
	for item in boiler:
		if (item.get('body') and item.get('title')) is not None:
			boilerplate.append((item.get('body')+ ' '+item.get('title')))
		elif (item.get('title') is None and item.get('body') is not None):
			boilerplate.append(item.get('body'))
		elif (item.get('title') is not None and item.get('body') is None):
			boilerplate.append(item.get('title'))
		else:
			boilerplate.append(None)
	print('append finish')
	return boilerplate

def clean_text(text):
	print('cleaning started')
	filtered_text=[]
	for t in text:
		if t is not None:
			tokens = get_tokens(t)
			filtered = [w for w in tokens if not (w in ignorechars or w in stopwords.words('english'))]
			filtered_text.append(filtered)
		else:
			filtered_text.append(None)
	return (filtered_text)

def list_to_string(l):
	print('start appending text')
	pool = []

	for text in l:
		if text is not None:
			words = ' '.join(text)
			pool.append(words)
		else:
			pool.append(None)
	return (pool)

def combine_column(col1,col2):
	all_combined=[]
	for x,y in col1,col2:
		if (x is not None or y is not None):
			combined=x+y
		elif(x is not None and y is None):
			combined=x
		elif(x is None and y is not None):
			combined=y
		else:
			combined=None
		all_combined.append(combined)
	return all_combined

def append_column(lst):
	se = pd.Series(lst)
	return (se.values)


# -- -- - -- --- - - -- MAIN PROGRAM -- -- - -- --- - - -- #

path=input('file path = ')
df = pd.read_csv(path)
print('read success')
body=df['body_3grams']
title=df['title_3grams']
#boiler = json_to_list('boilerplate')
#print('checking sample content: ')
#print(boiler[0], '\n')
#boilerplate=get_json_params(boiler)
#boiler_text=df['boiler_text']
#filtered_text = clean_text(boilerplate)
boiler_text = combine_column(body,title)
stemmed_text = get_stemmed(boiler_text)
boiler_text = list_to_string(stemmed_text)

# append column to dataframe
se_boiler=pd.Series(boiler_text)
df['boiler_2grams']=se_boiler.values
print(df.head(n=5))
save = input('save df? (Y/N)')
if save == 'Y':
	filename=input('filename = ')
	df.to_csv(filename, sep='\t')

