import pandas as pd
from textblob import TextBlob
import json
import nltk
import string
import math
from nltk.corpus import stopwords



# constant
ignorechars = [',', '.','-','--', '&', ';', ':', '?','#','>','<','=','[',']',')','(', '|','com','html', '!']

# functions
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

                
def append_column(lst):
	se = pd.Series(lst)
	return (se.values)


# -- -- - -- --- - - -- MAIN PROGRAM -- -- - -- --- - - -- #


df = pd.read_csv("test-stumble_upon.tsv", sep="\t")
print('read success')

boiler = json_to_list('boilerplate')
print('checking sample content: ')
print(boiler[0], '\n')
boilerplate=get_json_params(boiler)

filtered_text = clean_text(boilerplate)
boiler_text = list_to_string(filtered_text)
# append column to dataframe
se_boiler=pd.Series(boiler_text)
df['boiler_text']=se_boiler.values
print(df.head(n=5))
save = input('save df? (Y/N)')
if save == 'Y':
	df.to_csv('test-with-boilertext.tsv', sep='\t')
	
