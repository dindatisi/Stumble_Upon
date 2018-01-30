import pandas as pd
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm

# files
f_train = 'train_stemmed.tsv'
f_test = 'test_stemmed.tsv'


def load_data(f_train, f_test):
	train_data = list(np.array(pd.read_table(f_train))[:,2])
	train_url = list(np.array(pd.read_table(f_train))[:,0])
	test_data = list(np.array(pd.read_table(f_test))[:,2])
	test_url = list(np.array(pd.read_table(f_test))[:,0])
	y = np.array(pd.read_table(f_train))[:,-1]

	#combine all columns
	x_all = train_data + train_url + test_data + test_url

	len_train = len(train_data)

	return x_all,y, len_train

def get_tfidf(x_all):
	tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
		analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
	print ("fitting pipeline")
	tfv.fit(x_all)
	print ("transforming data")
	x_all = tfv.transform(x_all)
	print('checking first ten words: \n', tfv.get_feature_names()[:10])
	return x_all

def build_model(x_all, len_train):
	log = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
		C=1, fit_intercept=True, intercept_scaling=1.0, 
		class_weight=None, random_state=None)
	# cross-validation
	x_train = x_all[:len_train]
  	x_test = x_all[len_train:]
	train_score=np.mean(cross_validation.cross_val_score(log, x_train, y.astype(float), cv=20, scoring='roc_auc'))
	print ("20 Fold CV Score: ",train_score)
	return x_train,x_test

def get_prediction(x_train, x_test, y):
	print ("training on full data")
	log.fit(x_train,y.astype(float))
	pred = log.predict_proba(x_test)[:,1]
	testfile = pd.read_csv('test_stemmed.tsv', sep="\t", na_values=['?'], index_col=1)
	pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
	return pred_df

def main():
	x_all,y,len_train = load_data(f_train,f_test)
	x_tfidf = get_tfidf (x_all)
	x_train, x_test, y = build_model(x_tfidf, len_train)
	isContinue = input('continue to testing (Y/N)?').upper()
	if isContinue == 'Y':
		pred_df = get_prediction(x_train,x_test,y)
		filename=input('input filename to save = ')
		pred_df.to_csv(filename)
		print ("file saved")
	else:
		pass

main()
