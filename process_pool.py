#! usr/bin/env python
#! -*- coding: utf-8 -*-
from multiprocessing import Pool
from multiprocessing import Lock, Manager
import subprocess
import os
import re
import sys
import sklearn
import math
import pickle
#from sklearn import linear.model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn import cross_validation
from sklearn import svm
import numpy as np
import scipy as sp
import scipy.sparse as sps
from score_measurement import scoreMeasurement
from score_measurement import result_collector
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from score_measurement import fScore
from sklearn.cross_validation import StratifiedKFold
from multiprocessing import Pool,Lock,Manager
from joblib import Parallel,delayed
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
def get_ensemble_results(log_regression_p,log_regression_l,test_mat_profile,test_mat,y_test,alpha):
	results_set = []
	picked_alpha = alpha
	predict_p = log_regression_p.predict_proba(test_mat_profile)
	predict_l = log_regression_l.predict_proba(test_mat)
	final_result = picked_alpha*predict_p + (1 - picked_alpha)*predict_l
	for i in np.arange(0.30,0.31,0.01):
		temp_results = final_result[:,1] > i 	#0.35 # previously it was 0.4
		results_set.append(1*temp_results)
#	_temp_final_result = final_result[:,1] > 0.30
#	_temp_final_result = 1*_temp_final_result
	return results_set #_temp_final_result #results_set

def build_classifier(matrix_profile,matrix_latest_tweets,mat_c_P,mat_c_L,test_label,test_label_c,model_index,kf,alpha):
	class_label = np.array(test_label)
	list_of_results = []
	matrix_profile = matrix_profile.tocsr()
	matrix_latest_tweets = matrix_latest_tweets.tocsr()
	logRegression_p = LogisticRegression()
	logRegression_l = LogisticRegression()
	multinomial_p = MultinomialNB()
	gaussian_p = GaussianNB()
	svm_p = svm.SVC(kernel = 'linear')
	#multinomial_nb = lienar_model
	print('i am inside of the k-fold validation\n')
#	for train_index,test_index in kf:
#X_train_p,X_test_p = matrix_profile[train_index],matrix_profile[test_index]
#		X_train_l,X_test_l = matrix_latest_tweets[train_index],matrix_latest_tweets[test_index]
#		Y_train,Y_test = class_label[train_index],class_label[test_index]
	logRegression_l.fit(matrix_latest_tweets,test_label)
	#logRegression_l.fit(X_tr,Y_train)
	logRegression_p.fit(matrix_profile,test_label)
	final_result = get_ensemble_results(logRegression_p,logRegression_l,mat_c_P,mat_c_L,test_label_c,alpha)
	#for elem in xrange(0,len(final_result)):
	#	print(str(test_label_c[elem])+' '+ str(final_result[elem]) )
	list_of_results.append(fScore(expecresultList_ = test_label_c, precresultList_ = final_result))
	return list_of_results

def grab_result(result_set_tup):
	#key_list = result_set_tup.keys()
	#print(str(key_list))
	result_set = result_set_tup[0]
	shuffle_ind = result_set_tup[1]
        l = result_set_tup[2]
	file_i = result_set_tup[3]
	result_list = result_set_tup[4]

	print('F1 score is : ' + str(result_set.F1)+"\n")
	print('P score is : ' + str(result_set.P) +'\n')
	l.acquire()
	write_result = open('data/avg_shuffle_output/4_result_sets'+str(file_i)+'.txt','a')
	write_result_set = open('data/avg_shuffle_output/7_result_sets' + str(file_i) + '.txt','a')
	write_output = open('data/avg_shuffle_output/error_output.txt','a')
	#write_result.write('shuffling iterations :' + str(shuffle_ind)+'\n' )
	write_result.write('best_combinations: ')
	write_result_set.write('best_combinations: ')
	#result_set_list = result_set.best_combination.split(' ')
	p = re.compile('[a-zA-Z]+')
	for each in xrange(0,len(result_set.best_combination)):
		temp_ob = p.match(result_set.best_combination[each])
		if(temp_ob == None):
			continue
		if(each != len(result_set.best_combination) - 1):
		    	write_result.write(result_set.best_combination[each]+',')
		else:
			write_result.write(result_set.best_combination[each] +' ')
	for each in xrange(0,len(result_list)):
		for sub in xrange(0,len(result_list[each].best_combination)):
			temp_ob = p.match(result_list[each].best_combination[sub])
			if(temp_ob == None):
				continue
			if(sub != (len(result_list[each].best_combination) - 1)):
				write_result_set.write(result_list[each].best_combination[sub] + ',')
			else:
				write_result_set.write(result_list[each].best_combination[sub] + ' ')
		write_output.write('i am writing the required output\n')
		write_result_set.write('current_alpha_score: ' + str(math.fabs(result_list[each].beta))+' ')
		write_result_set.write('best_F1: ' + str(result_list[each].F1)+' ')
		write_result_set.write('best_precision: ' + str(result_list[each].P)+' ')
		write_result_set.write('best_recall: ' + str(result_list[each].R) + '\n')
    	write_result.write('current_alpha_score: ' + str(math.fabs(result_set.beta))+' ')
    	write_result.write('best_F-1: ' + str(result_set.F1)+' ')
    	write_result.write('best_precision: ' + str(result_set.P)+' ')
    	write_result.write('best_recall: ' + str(result_set.R)+'\n')

	result_set.F1 = 0
	write_result.close()
	write_result_set.close()
	write_output.close()
	l.release()
	return
def work_with_combinations(list_of_combinations,y,y_c,p_mat,l_mat,p_mat_c,l_mat_c,global_data_mat,beta_val,shuffle_index,l,buildingClassifier,kf,output_file_index):
	result_set = result_collector()
	print('execute now :')
	result_lists = []
	for each in list_of_combinations:
		print(each)
    		temp_sparse_P_mat = sps.csr_matrix(p_mat,copy=True)
		temp_sparse_L_mat = sps.csr_matrix(l_mat,copy = True)
		temp_sparse_P_mat_c = sps.csr_matrix(p_mat_c,copy = True)
		temp_sparse_L_mat_c = sps.csr_matrix(l_mat_c,copy = True)
		try:
    			for sub_each in each:
				print('each element :' + sub_each)
				#print(str(global_data_mat[sub_each].shape))
				if(sub_each.find('pro') != -1):
    					temp_sparse_P_mat = hstack([temp_sparse_P_mat,global_data_mat[sub_each.strip()+ '_train' ]])
					temp_sparse_P_mat_c = hstack([temp_sparse_P_mat_c,global_data_mat[sub_each.strip() + '_cross']])
				else:
					temp_sparse_L_mat = hstack([temp_sparse_L_mat,global_data_mat[sub_each.strip() + '_train' ]])
					temp_sparse_L_mat_c = hstack([temp_sparse_L_mat_c,global_data_mat[sub_each.strip() + '_cross']])
		except Exception as e:
			print(str(e))
		print('start of this training and testing')
		list_of_result = build_classifier(temp_sparse_P_mat,temp_sparse_L_mat,temp_sparse_P_mat_c,temp_sparse_L_mat_c,y,y_c,1.5,kf,beta_val)
		print('i am breaking down here i guess')
    		fscore_result_F1 = []
		fscore_result_P = []
		fscore_result_R = []
	        #print(str(list_of_result))
    		for sub_sub_each in list_of_result:
			F1,P,R = scoreMeasurement(sub_sub_each,1)
    			fscore_result_F1.append(F1)
			fscore_result_P.append(P)
			fscore_result_R.append(R)
		p,r,f1 = 0,0,0
		if(len(fscore_result_F1) == 0 ):
			print('continue')
			continue
		if(result_set.F1 < float(sum(fscore_result_F1))/len(fscore_result_F1)):
			result_set.beta = beta_val
			result_set.best_combination = each[:]
			print(str(result_set.best_combination))
			result_set.F1 = float(sum(fscore_result_F1))/len(fscore_result_F1)
			result_set.P = float(sum(fscore_result_P))/len(fscore_result_P)
        		result_set.R =  float(sum(fscore_result_R))/len(fscore_result_R)
		if((0.90 - (float(sum(fscore_result_F1))/len(fscore_result_F1))) <= 0.01 and (0.90 - (float(sum(fscore_result_F1))/len(fscore_result_F1))) > 0):
			print('i am inside multiple execution point\n')
			result_set_component = result_collector()
			result_set_component.beta = beta_val
			result_set_component.best_combination = each[:]
			result_set_component.F1 = float(sum(fscore_result_F1))/len(fscore_result_F1)
			result_set_component.P = float(sum(fscore_result_P))/len(fscore_result_P)
			result_set_component.R = float(sum(fscore_result_R))/len(fscore_result_R)
			result_lists.append(result_set_component)
			print('end of collection\n')

		print('average f1 score ' + str(float(sum(fscore_result_F1))/len(fscore_result_F1)))
        	print('average P score ' + str(float(sum(fscore_result_P))/len(fscore_result_P)))
       		print('average R score ' + str(float(sum(fscore_result_R))/len(fscore_result_R)))
		#del temp_sparse_mat
	return (result_set, shuffle_index, l, output_file_index,result_lists)
def get_kf_result(list_of_result,each_val,beta_val):
	fscore_result_F1 = []
	fscore_result_P = []
	fscore_result_R = []
	result_set = result_collector()
	#print(str(list_of_result))
    	for sub_sub_each in list_of_result:
		F1,P,R = scoreMeasurement(sub_sub_each,1)
		print(str(F1) +' '+ str(P) +' ' +str(R))
    		fscore_result_F1.append(F1)
		fscore_result_P.append(P)
		fscore_result_R.append(R)
		p,r,f1 = 0,0,0
		if(len(fscore_result_F1) == 0 ):
			print('continue')
			continue
	#if(result_set.F1 < float(sum(fscore_result_F1))/len(fscore_result_F1)):
	result_set.beta = beta_val
	result_set.best_combination = each_val[:]
	print(str(result_set.best_combination))
	result_set.F1 = float(sum(fscore_result_F1))/len(fscore_result_F1)
	result_set.P = float(sum(fscore_result_P))/len(fscore_result_P)
        result_set.R =  float(sum(fscore_result_R))/len(fscore_result_R)

	return result_set

def do_map_operation(X_train_p,X_test_p,X_train_l,X_test_l,Y_train,Y_test,alpha,test_index):
	finding_list = ['ecig','e-cig','vapo','vaping','ejuice','eliquid','vapes']
	logRegression_p = LogisticRegression()
	logRegression_l = LogisticRegression()
	multinomial_p = MultinomialNB()
	gaussian_p = GaussianNB()
	svm_p = svm.SVC(kernel = 'linear')
	logRegression_p.fit(X_train_p,Y_train)
	logRegression_l.fit(X_train_l,Y_train)
	print('we have created 5 folds here\n')
	final_result = get_ensemble_results(logRegression_p,logRegression_l,X_test_p,X_test_l,Y_test,alpha)
	#get_result.append(fScore(expecresultList_ = Y_test, precresultList_ = final_result))
	'''for elem in xrange(0,len(final_result)):
		if(final_result[elem] == 0):
			for each in finding_list:
				if(temp_user_name_list[elem].lower().find(each)!= -1):
					final_result[elem] = 1
					break'''
	list_of_results = []
	for each in final_result:
		list_of_results.append(fScore(expecresultList_ = Y_test,precresultList_ = each, test_index_ = test_index))
	return list_of_results#fScore(expecresultList_ = Y_test, precresultList_ = final_result, test_index_ = test_index)

def do_map_operations(single_arg):
	return do_map_operation(*single_arg)
def with_shuffle_operation(global_mat,base_mat_p,base_mat_l,class_label,feature_list,alpha,kf_l,indexing,user_name_list):
	shuffle_index = 0
	with open('data/avg_shuffle_output/with_username_features1001_'+str(indexing)+'.txt','a+') as data_writer:
		for shuffle_index in xrange(0,500): # previously it was 500 for creating shuffles
			temp_sparse_mat_p = sps.csr_matrix(base_mat_p,copy = True)
			temp_sparse_mat_l = sps.csr_matrix(base_mat_l,copy = True)
			#temp_base_mat_test_p = sps.csr_matrix(base_mat_train,copy = True)
			#temp_base_mat_test_l = sps.csr_matrix(base_mat_test,copy = True)
			labels = class_label[:]
			temp_user_name_list = user_name_list[:]
			if(len(feature_list) != 0 ):
				for each in feature_list:
					if(each.find('pro') != -1):
						temp_sparse_mat_p = hstack([temp_sparse_mat_p,global_mat['stra_' + each + '_kfold']])
						#temp_sparse_mat_p = hstack([temp_sparse_mat_p,global_mat['stra_' + each + '_train']]) # this is for 70 percent training set data
						#temp_base_mat_test_p = hstack([temp_base_mat_p,global_mat['stra_' + each + '_test']])
					else:
						temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + each +'_kfold']])
						#temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + each + '_train']) # this is for 70 percent training sets data
						#temp_base_mat_test_l = hstack([temp_base_mat_l,global_mat['stra_' + each +'_test' ]])
			temp_sparse_mat_p,temp_sparse_mat_l,labels,temp_user_name_list = shuffle(temp_sparse_mat_p,temp_sparse_mat_l,labels,temp_user_name_list) # i dont want to shuffle now thanks
			#list_items = [(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],class_label[train_index],class_label[test_index],float(alpha)) for train_index, test_index in kf_l]
			#temp_sparse_mat_p = sps.csr_matrix(base_mat_p)# to ensure that the matrices are csr matrix instead of coo matrix
			#temp_sparse_mat_l = sps.csr_matrix(base_mat_l)
			get_result  = Parallel(n_jobs = 5)(delayed(do_map_operation)(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],labels[train_index],labels[test_index],float(alpha),test_index) for train_index, test_index in kf_l)
			#get_result = p.map(do_map_operations,list_items)
			ultimate_result = []
			for datum in get_result:
				ultimate_result.append(get_kf_result(datum,feature_list,float(alpha)))#get_kf_result(get_result,feature_list,float(alpha))
			s_p,s_r,s_f1 = 0,0,0
			for each in ultimate_result:
				s_p += each.P
				s_r += each.R
				s_f1 += each.F1
			data_writer.write('best_F_1: '+ str(s_f1/len(ultimate_result)) + ' ')
			data_writer.write('best_precision: ' + str(s_p / len(ultimate_result)) + ' ')
			data_writer.write('best_recall: '  + str(s_r/len(ultimate_result)) + ' ' )
			data_writer.write('\n')


	return
def work_with_folds(global_mat,base_mat_p,base_mat_l,list_of_features,class_labels,kf,alpha,indexing = None,user_name_list = None):
	ultimate_result = result_collector()
	kf_l = cross_validation.StratifiedKFold(class_labels,n_folds = 5)
	get_result = []
	temp_sparse_mat_p = sps.csr_matrix(base_mat_p,copy = True)
	temp_sparse_mat_l = sps.csr_matrix(base_mat_l, copy = True)
	for each in list_of_features:
		if(each.find('pro') != -1):
			temp_sparse_mat_p = hstack([temp_sparse_mat_p ,global_mat['stra_'+ each + '_train'] ])
		else:
			temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + each + '_train']])
	temp_sparse_mat_p = sps.csr_matrix(temp_sparse_mat_p)
	temp_sparse_mat_l = sps.csr_matrix(temp_sparse_mat_l)		
	temp_sparse_mat_p,temp_sparse_mat_l,class_labels = shuffle(temp_sparse_mat_p,temp_sparse_mat_l,class_labels) # i dont want to shuffle now thanks

	get_result_sets = Parallel(n_jobs = 5)( delayed(do_map_operation)(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],class_labels[train_index],class_labels[test_index],float(alpha),test_index) for train_index, test_index in kf_l)
	for datum_index in xrange(0,len(get_result_sets[0])):
		sum_of_p,sum_of_f1,sum_of_r = 0,0,0
		for sub_datum_index in xrange(0,len(get_result_sets)):
			q,p,r = scoreMeasurement(get_result_sets[sub_datum_index][datum_index],1)
			sum_of_p += p
			sum_of_f1 += q
			sum_of_r += r
		sum_of_p = sum_of_p/len(get_result_sets)
		sum_of_f1 = sum_of_f1/len(get_result_sets)
		sum_of_r = sum_of_r/len(get_result_sets)
		#print('with threshold value: ' + str(datum_index))
	        print('best_precision: ' + str(sum_of_p) +' '+ 'best_recall: ' + str(sum_of_r) + 'best_f1: ' + str(sum_of_f1) )
	return
def parse_resultant_output(list_of_combinations,global_mat,output_path,base_mat_p,base_mat_l,class_label,kf,alpha):
	ultimate_result = result_collector()
	kf_l = cross_validation.StratifiedKFold(class_label,n_folds = 5) 
	get_result = []
	p = Pool(processes = 5)
	print(str(base_mat_p.shape))
	print(str(base_mat_l.shape))
	with  open('data/avg_shuffle_output/19_x_validation_result_set.txt','a+') as data_writer:
	#for data in list_of_combinations:
		#data = data_reader.readlines()
		for each in list_of_combinations:
			'''temp_list = each.split('best_F1')[0]
			alpha = temp_list.split('current_alpha_score:')[1].strip()
			feature_list = temp_list.split('current_alpha_score:')[0]
			main_feature_list = feature_list.split(',')#[1:]'''
			temp_sparse_mat_p = sps.csr_matrix(base_mat_p,copy = True)
			temp_sparse_mat_l = sps.csr_matrix(base_mat_l,copy = True)
			#list_items = [(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],class_label[train_index],class_label[test_index],float(alpha)) for train_index, test_index in kf]

			for sub in each:
				if(sub.find('pro') != -1):
					temp_sparse_mat_p = hstack([temp_sparse_mat_p, global_mat['stra_' + sub.strip(' ') +'_train']])
				else:
					temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + sub.strip(' ') + '_train']])
			counter = 0
			temp_sparse_mat_p = sps.csr_matrix(temp_sparse_mat_p)
			temp_sparse_mat_l = sps.csr_matrix(temp_sparse_mat_l)
			#list_items 
			list_items = [(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],class_label[train_index],class_label[test_index],float(alpha)) for train_index, test_index in kf]

			#get_result  = Parallel(n_jobs = 5)(delayed(do_map_operation)(temp_sparse_mat_p[train_index],temp_sparse_mat_p[test_index],temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index],class_label[train_index],class_label[test_index],float(alpha)) for train_index, test_index in kf_l)
			get_result = p.map(do_map_operations,list_items)
			'''for train_index,test_index in kf:
				#print(str(train_index) +' '+ str(test_index))
				#counter += 1 
				#print(str(counter))
				X_train_p = temp_sparse_mat_p[train_index] 
				X_test_p = temp_sparse_mat_p[test_index]
				X_train_l,X_test_l = temp_sparse_mat_l[train_index],temp_sparse_mat_l[test_index]
				Y_train,Y_test = class_label[train_index],class_label[test_index]
				logRegression_p = LogisticRegression()
				logRegression_l = LogisticRegression()
				multinomial_p = MultinomialNB()
				gaussian_p = GaussianNB()
				svm_p = svm.SVC(kernel = 'linear')
				logRegression_p.fit(X_train_p,Y_train)
				logRegression_l.fit(X_train_l,Y_train)
				final_result = get_ensemble_results(logRegression_p,logRegression_l,X_test_p,X_test_l,Y_test,float(alpha))
				get_result.append(fScore(expecresultList_ = Y_test, precresultList_ = final_result))'''
			
			ultimate_result = get_kf_result(get_result,ultimate_result,each,float(alpha))
		data_writer.write('best_combinations: ')
		for each_elem in ultimate_result.best_combination:
			data_writer.write( each_elem +', ')
		data_writer.write('alpha_value: ' + str(ultimate_result.beta))
		data_writer.write('best_F_1:'+ str(ultimate_result.F1) + ' ')
		data_writer.write('best_precision: ' + str(ultimate_result.P) + ' ')
		data_writer.write('best_recall: '  + str(ultimate_result.R) + '\n' )
			#p.join()
			#p.close()

	return
def train_model(global_mat,base_mat_p,base_mat_l,class_label,feature_list,alpha):
	
	temp_sparse_mat_p = sps.csr_matrix(base_mat_p)
	temp_sparse_mat_l = sps.csr_matrix(base_mat_l)
	logRegression_p = LogisticRegression()
	logRegression_l = LogisticRegression()
	for each in feature_list:
		if(each.find('pro') != -1):
			temp_sparse_mat_p = hstack([temp_sparse_mat_p,global_mat['stra_' + each.strip() + '_train']])
		else:
			temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + each.strip() + '_train']])
	logRegression_p.fit(temp_sparse_mat_p,class_label)
	logRegression_l.fit(temp_sparse_mat_l,class_label)
	with  open('data/model/classifier_model_p1.1','wb') as model_p,open('data/model/classifier_model_l1.1','wb') as model_l:
		pickle.dump(logRegression_p,model_p,pickle.HIGHEST_PROTOCOL)
		pickle.dump(logRegression_l,model_l,pickle.HIGHEST_PROTOCOL)
		
	return
def test_model(global_mat,base_mat_p,base_mat_l,class_label,feature_list,alpha):

	with open('data/model/classifier_model_p1.1','rb') as model_p, open('data/model/classifier_model_l1.1','rb') as model_l:
		clf_p = pickle.load(model_p)
		clf_l = pickle.load(model_l)
		print(str(base_mat_p.shape)+'\n')
		print(str(base_mat_l.shape)+'\n')
		base_mat_test_p = sps.csr_matrix(base_mat_p)
		base_mat_test_l = sps.csr_matrix(base_mat_l)
		for each_elem in feature_list:
			print(str(each_elem) +' ')
			print(str(global_mat[each_elem +'_test'].shape))
			if(each_elem.find('pro') != -1):
				base_mat_test_p = hstack([base_mat_test_p, global_mat[each_elem+'_test']])
			else:
				base_mat_test_l = hstack([base_mat_test_l,global_mat[each_elem+'_test']])
		final_result = get_ensemble_results(clf_p,clf_l,base_mat_test_p,base_mat_test_l,class_label,alpha)
		for elem in xrange(0,len(final_result)):
		#	print(str(test_label_c[elem])+' '+ str(final_result[elem]) )
			f_ob = fScore(expecresultList_ = class_label, precresultList_ = final_result[elem])
			F1,P,R = scoreMeasurement(f_ob,1)
			print('test F1: ' + str(F1) +' '+'test P: ' + str(P) +' '+'test R: ' + str(R))
		
	return
def train_test_operations(global_mat,base_mat_p,base_mat_l,class_label,feature_list,alpha):
	temp_sparse_mat_p = sps.csr_matrix(base_mat_p,copy = True)
	temp_sparse_mat_l = sps.csr_matrix(base_mat_l,copy = True)
	#temp_base_mat_test_p = sps.csr_matrix(base_mat_train,copy = True)
	#temp_base_mat_test_l = sps.csr_matrix(base_mat_test,copy = True)
	labels = class_label[:]
	for each in feature_list:
		if(each.find('pro') != -1):
			temp_sparse_mat_p = hstack([temp_sparse_mat_p,global_mat['stra_' + each + '_kfold']])
			#temp_base_mat_test_p = hstack([temp_base_mat_p,global_mat['stra_' + each + '_test']])
		else:
			temp_sparse_mat_l = hstack([temp_sparse_mat_l,global_mat['stra_' + each +'_kfold']])
			#temp_base_mat_test_l = hstack([temp_base_mat_l,global_mat['stra_' + each +'_test' ]])
	count = 0
	while(True):
		if(count == 1): break
		temp_sparse_mat_p_r,temp_sparse_mat_l_r,labels_r = temp_sparse_mat_p,temp_sparse_mat_l,labels
		count += 1
		temp_sparse_mat_p_r_tr,temp_sparse_mat_p_r_te,temp_sparse_mat_l_r_tr,temp_sparse_mat_l_r_te,labels_r_tr,labels_r_te = train_test_split(temp_sparse_mat_p_r,temp_sparse_mat_l_r,labels_r,test_size = 0.40,random_state = 42)
		clf_p = LogisticRegression()
		clf_l = LogisticRegression()
		clf_p.fit(temp_sparse_mat_p_r_tr,labels_r_tr)
		clf_l.fit(temp_sparse_mat_l_r_tr,labels_r_tr)
		final_result = get_ensemble_results(clf_p,clf_l,temp_sparse_mat_p_r_te,temp_sparse_mat_l_r_te,labels_r_te,alpha)
		# for elem in xrange(0,len(final_result)):
		# print(str(test_label_c[elem])+' '+ str(final_result[elem]) )
		for each_elem in final_result:
			f_ob = fScore(expecresultList_ = labels_r_te, precresultList_ = each_elem)
			F1,P,R = scoreMeasurement(f_ob,1)
			print('test_F1: ' + str(F1) + ' ' + 'test_P: ' + str(P) + ' ' + 'test_R: ' + str(R))

	return
