import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_text_array(boiler_text):
	boilerplate=[]
	print('getting rid of Null values')
	for text in boiler_text:
		if text is not None:
			boilerplate.append(text)
		else:
			pass
	# Create text
	print('create text')
	text_data = np.array(boilerplate)
	return text_data

def get_tfidf_matrix(text_list):
	tfidf = TfidfVectorizer()
	feature_matrix = tfidf.fit_transform(text_list)
	print(feature_matrix)
	return (tfidf.get_feature_names(), feature_matrix)

def split_set(df):
	print('split train/test')
	msk = np.random.rand(len(df)) < 0.75
	train_75 = df[msk]
	test_25 = df[~msk]
	return (train_75, test_25)


# MAIN
df = pd.read_csv("train-with-boilertext.tsv", sep="\t")
df.head()

boiler_text=df['boiler_text']
boiler_text=get_text_array(boiler_text)
boiler_list=boiler_text.tolist()
print('list length= ', len(boiler_list))

# Create the tf-idf feature matrix
word_list, matrix=get_tfidf_matrix(boiler_list)
tfidf_arr = matrix.toarray()
np.round_(tfidf_arr, decimals=4)
print(tfidf_arr.shape)
print('total words= ', len(word_list))

#create new df for tfidf
df_tf=pd.DataFrame(data=tfidf_arr)
df_tf.columns=word_list
df_tf['urlid']=df['urlid']
df_tf.head()

print('start merging')
merged_df=pd.merge(df, df_tf, how='inner', on='urlid',
         left_index=False)
print('merge success')
print('remove unnecessary columns')
merged_df=merged_df.drop('boilerplate',1)
merged_df=merged_df.drop('boiler_text',1)
print(merged_df.columns)

isSplit=input('do u want to split data? (Y/N)')
if isSplit=='Y':
	print('splitting data')
	train, test=split_set(merged_df)
	print('len train: %d , len test: %d' %(len(train), len(test)))
	save = input('save df? (Y/N)')
	if save == 'Y':
		print('wait to save training set')
		train.to_csv('train75_tfidf.tsv', sep='\t')
		print('save successful')

		print('wait to save testing set')
		test.to_csv('train_test25_tfidf.tsv', sep='\t')
		print('save successful')
	else:
		print('data notsaved')
elif isSplit=='N':
	print('wait to save data')
	merged_df.to_csv('test_tfidf.tsv', sep='\t')
	print('save successful')


