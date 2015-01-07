#! usr/bin/env python
#! -*- coding: utf-8 -*-
# this is for reading the urls
import re
import os
import random
import subprocess
import numpy as np
import scipy as sp
dictUrls = {}

class dataObjectContainers(object):
    def __init__(self,X= None,y=None,y_rank = None,latest_tweets = None,user_names=None):
        self.X = X
        self.y = y
        self.y_rank = y_rank
        self.latest_tweets = latest_tweets
        self.hash_tag = None
	self.user_name_list = user_names
        return

def startOperatingandLoad():
    urls_vocabulary = list()
    global dictUrls
    for i in range(0,6):
        file_object_urls = open('data/urlsuExpandedOutput'+str(i)+'.txt')
        try:
            data = file_object_urls.readlines()
        finally:
            file_object_urls.close()
        urls_vocabulary.extend(data)
        #print(len(urls_vocabulary))
    
    for dat in urls_vocabulary:
        stringVal = dat.split("\t")     
        if(len(stringVal) == 5 or len(stringVal) == 4):
            stringVal[2].strip()
            dictUrls[stringVal[2]] =stringVal[3].strip()
            #print(dictUrls[stringVal[2]])
        else:
            dictUrls[stringVal[2]] = None;
    return

def read_urls_from_source(source_path):
	dict_of_urls = {}
	file_to_read_urls = open(source_path,'r+')
	try:
		data = file_to_read_urls.readlines()
	finally:
		file_to_read_urls.close()
	for datum in data:
		sub_datum = datum.split('\t')
		#sub_sub_datum = sub_datum[2].split(' ') 
		if(dict_of_urls.get(sub_datum[0],None) == None):
			dict_of_urls[sub_datum[0]] = set()
		#for each in sub_sub_datum:
		dict_of_urls[sub_datum[0]].add(sub_datum[4])
			
	return dict_of_urls
#this is for parsing the data
def find_urls(username,tweets):
	url_list = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweets)
	return url_list

def parse_data(data):
    X = []
    y = []
    match_string = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    user_name_list = []
    count_pos = 0
    global dictUrls
    get_dict_urls = read_urls_from_source('data/expandedincompleteUrls0.txt')
    for datum in data:
        list_data = datum.split('\t')
	url_list = find_urls(list_data[1],list_data[2])
	#print('current position' + str(count_pos))     
	#list_data[3] = list_data[3].replace('rt','')
	data_list = list_data[2].split(' ')
	temp_data_list = data_list[:]
	if(len(url_list) != 0):
		for elem in url_list:
			temp_data_list = [s for s in temp_data_list if s not in elem]
	temp_data_holder = ' '.join(temp_data_list)
	url_list = []
	# X.append(temp_data_holder)#list_data[3].decode('utf-8','ignore'))
	if(get_dict_urls.get(list_data[1],None) != None):
		for element in get_dict_urls[list_data[1]]:
			temp_data_holder = temp_data_holder+' '+element
			url_list.append(element)
	else:
	 	if(dictUrls.get(list_data[1],None) != None):
			for each_elem in dictUrls[list_data[1]]:
				temp_data_holder = temp_data_holder +' '+ each_elem
				url_list.append(each_elem)
	i = 0
	temp_datas = list_data[2]
	for m in re.finditer(match_string,list_data[2]):
		if(i == len(url_list)): break
		temp_datas = re.sub(match_string,url_list[i],temp_datas)
		i += 1
	X.append(temp_datas.decode('utf-8','ignore'))#temp_data_holder.decode('utf-8','ignore'))
	#print(temp_data_holder)
	user_name_list.append(list_data[1].decode('utf-8','ignore'))
        #print('data is '+ list_data[2])
        count_pos += 1
        #print('pos is ' + str(count_pos))
        y.append(int(list_data[0].strip()))
    data_object = dataObjectContainers(X = X,y=y,user_names = user_name_list)
    return data_object
def read_data_source(path_name):
	file_ob = open(path_name,'r+')
	try:
		data = file_ob.readlines()
	finally:
		file_ob.close()
	return data
def work_with_pos(data_list = None,file_path = None):

	data_source = read_data_source('data/latest_tweets_testing_set.txt')
	#tuples = parse_data(data_source) # it was previously used
        file_to_write = open('data/ecig_latest_tweets_pos_testing.txt','w+')
	index = 0
	dict_of_tags = {}
	'''for each in tuples.X:
		file_to_write = open('Binary_classifier_for_health/dir_health/health_dataset_pos'+str(index)+'.txt','w+')
		file_to_write.write(each)
		file_to_write.close()
		index += 1'''
	list_of_data = []
	for each in xrange(0,len(data_source)):#len(tuples.X)):
		argsList = ['../../Binary_classifier_for_health/ark-tweet-nlp-0.3.2/runTagger.sh',"--output-format",'conll', 'data/tweet_data_files/source_'+str(each)+'.txt']
		temp_data = subprocess.Popen(argsList,stdout=subprocess.PIPE).communicate()
		#print argsList
		#print temp_data
		temp_data = temp_data[0]
		#print(temp_data)
		list_of_data.append(temp_data.split('\n'))
		#print('size of list '+str(len(temp_data.split('\n'))))
	'''for each in list_of_data:
		for sub_each in each:
			print(sub_each)'''
	get_data_set = []
	index = 0
	for each in list_of_data:
	 	temp_string = ''
		for sub_each in each:
			if(len(sub_each.split('\t')) >= 2):
				temp_list_data = sub_each.split('\t')
				if(dict_of_tags.get(temp_list_data[1],None) == None):
					dict_of_tags[temp_list_data[1]] = index 
					index += 1
				word_pos = temp_list_data[0] +'_'+ temp_list_data[1]
				temp_string = temp_string + word_pos +' '
		file_to_write.write(temp_string+'\n')
		get_data_set.append(temp_string.strip())
	'''file_for_pos_tags = open('Binary_classifier_for_health/dir_health/pos_tag.txt','w+')
	for (key,val) in dict_of_tags.items():
		file_for_pos_tags.write(key+'\t'+str(val)+'\n')
	file_for_pos_tags.close()
	print(str(index)+'\n')'''
	file_to_write.close()
	
	'''for each in xrange(0,index):#len(tuples.X))
		temp_data = os.system('Binary_classifier_for_health/ark-tweet-nlp-0.3.2/./runTagger.sh --output-format conll Binary_classifier_for_health/dir_health/health_dataset_pos'+ str(each)+'.txt')		  
 		print(type(temp_data))'''
	dict_pos = {}
	return (get_data_set,dict_of_tags)
def get_pos_tags(path_name):
	#data_set , dict_for_tags = work_with_pos()
	read_pos_tags = open('data/pos_tag_index.txt')
	read_data = open(path_name)
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
        dict_for_tags = {}
	for each in data_pos_tags:
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = int(temp_str[1].strip())
        pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	for index , each in enumerate(data_set):
		temp_each = each.split(' ')
		for sub_each in temp_each:
			temp_str = sub_each[sub_each.rfind('_') + 1 :].rstrip()
			#print(temp_str)
			if(dict_for_tags.get(temp_str,None) != None):
				#print(str(dict_for_tags.get(temp_str,None)))
				pos_tag_mat[index][dict_for_tags[temp_str]] += 1
	#print(pos_tag_mat)
	return pos_tag_mat
def get_consecutive_pos_tag(path_name):
	
	read_pos_tags = open('data/pos_tag_index.txt')
	read_data = open(path_name)
	consecutive_pos_tag_elem = []
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
        dict_for_tags = {}
	for each in data_pos_tags:
		if(len(each) == 0):
			continue
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = int(temp_str[1].strip())
        pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	data_set_len = len(data_set)
	for index , each in enumerate(data_set):
		temp_each = each.split(' ')
		temp_elem = ''
		#print(temp_each)
		for i in xrange(0,len(temp_each)-2):
			temp_str_1 = temp_each[i][temp_each[i].rfind('_') + 1 :]
			for j in xrange(i + 1,i + 3):
				temp_str_1 += temp_each[j][temp_each[j].rfind('_') + 1 :]
			temp_elem = temp_elem + temp_str_1 +' '
		consecutive_pos_tag_elem.append(temp_elem)

	return consecutive_pos_tag_elem # it will  return list of consecutive pos_tags for all tweets
def tokens_with_tags():
	read_pos_tags  = open('data/pos_tag_index.txt')
	read_data = open('data/ecig_dataset_pos.txt')
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
	dict_for_tags = {}
	for each in data_pos_tags:
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = temp_str[1]
	pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	vectorizer = CountVectorizer(min_df= 1,stop_words='english',ngram_range=(1,2))
	mat = vectorizer.fit_transform(data_pos_tags)
	return mat
#def start_main_func():
#	work_with_pos()
#	return
#start_main_func()
