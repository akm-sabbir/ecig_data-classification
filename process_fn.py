#! usr/bin/env python
#! -*- codinng: utf-8 -*-

import os
import re
import sys

def read_file(path_name):
	file_reader = open(path_name,'r+')
	try:
		data = file_reader.readlines()
	finally:
		file_reader.close()

	return data

def main_operation():
	data = read_file('data/avg_shuffle_output/test_indexing.txt')
	source_data = read_file('data/training_set.txt')
	source_data.extend(read_file('data/cross_validation_set.txt'))
	source_data.extend(read_file('data/testing_set.txt'))
	latest_tweets = read_file('data/latest_tweets_training_set.txt')
	latest_tweets.extend(read_file('data/latest_tweets_cross_validation_set.txt'))
	latest_tweets.extend(read_file('data/latest_tweets_testing_set.txt'))
	print(str(len(data))+'\n')
	with open('data/avg_shuffle_output/fn_data.txt','w+') as writer:
		for each in data:
			each = int(each)
			cl,un,profile_des = source_data[each].split('\t')
			lt = latest_tweets[each].split('\t')
			print(cl+' '+un + ' ' + profile_des)
			writer.write(cl + '\t' + un + '\t' + profile_des +'\t' + 'latest_tweets_info: '+ '\t')
			for datum in xrange(0,min(20,len(lt))):
				writer.write(lt[datum].strip('\n')+'\t')
			writer.write('\n')
	
	return
main_operation()
