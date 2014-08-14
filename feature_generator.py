#! usr/bin/env/ python
#! -*- coding: utf-8 -*-
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle
def storeModels(clf,i_i,t_t):
    if( t_t == 'c'):
        model_saver = open("data/linear_models/model_ecig"+str(i_i),"wb")
    else:
        model_saver = open("data/linear_models/vectorizer_ecig"+str(i_i),"wb")
    pickle.dump(clf, model_saver, pickle.HIGHEST_PROTOCOL)
    #model_saver.close()
    return

def generates_features(X,version):
    vectorizer = CountVectorizer(min_df = 1,stop_words='english',ngram_range = (1,3))
    matrix = vectorizer.fit_transform(X)
    matrix = matrix.astype('double')
    #matrix = Normalizer(copy=False).fit_transform(matrix)
    # print(str(matrix.shape))
    # print(str(len(vectorizer.get_feature_names())))
    storeModels(clf = vectorizer, i_i =version, t_t = 'v')
    # print(str(matrix.shape))
    return matrix


def generates_features_transform(X,version,model_type):
	if(model_type == 'c'):
		path_name = 'data/linear_models/model_ecig'
	else:
	 	path_name ='data/linear_models/vectorizer_ecig'
	model_retrieve  = open(path_name + str(version),'rb')
	data_model = pickle.load(model_retrieve)
	mat = data_model.transform(X)
	return mat
