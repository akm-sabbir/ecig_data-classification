#! /usr/bin/env python
#! -*- coding- utf-8 -*-

import csv,collections
import nltk
import os
import re
import numpy as np
import scipy as sp
def load_sent_word_net(format_specifier = None):
	sent_scores = collections.defaultdict(list)
	sent_scores_without_pos = collections.defaultdict(list)
	format_ = format_specifier if format_specifier != None else "/"
	with open(os.path.join("data/","SentiWordNet_3.0.0_20130122.txt"),"r+") as csvfile:
		reader = csv.reader(csvfile,delimiter ='\t',quotechar='"')
		index = 0
		for line in reader:
			if(line[0].startswith("#")):
				continue
			if(len(line) == 1):
				continue
			POS,ID,PosScore,NegScore,SynsetTerms,Gloss = line
			#print(str(PosScore)+' ' +str(NegScore))
			#index += 1
			#print(str(index))
			for term in SynsetTerms.split(" "):
				term = term.split("#")[0]
				term = term.replace("-"," ").replace("_"," ")
				key = ("%s"+ format_ +"%s")%(POS,term.split("#")[0].lower())
				sent_scores[key].append((float(PosScore),float(NegScore)))
				sent_scores_without_pos[term.split("#")[0].lower()].append((float(PosScore),float(NegScore)))
	for key,value in sent_scores.iteritems():
		sent_scores[key] = np.mean(value,axis = 0)
	for key,value in sent_scores_without_pos.iteritems():
		sent_scores_without_pos[key] = np.mean(value,axis = 0)
	return sent_scores,sent_scores_without_pos
def using_nltk_get_avg_score(data,sent_scores_,sent_scores_without_,format_):
	temp_data = tuple(data.split(' '))
	pos_tagged = nltk.pos_tag(temp_data)
	pos_score = []
	neg_score = []
	pos_score_without = []
	neg_score_without = []
	noun = 0
	adjective = 0
	verb = 0
	adverb = 0
	for w,pt in pos_tagged:
		p,n =0,0
		p_w,n_w = 0, 0
		sent_pos_type = None
		if(pt.startswith("NN")):
			noun += 1
			sent_pos_type = "n"
		elif(pt.startswith("JJ")):
			adjective += 1
			sent_pos_type = "a"
		elif(pt.startswith("VB")):
			verb += 1
			sent_pos_type = "v"
		elif(pt.startswith("RB")):
			adverb += 1
			sent_pos_type = "r"
		sent_word = None
		if(sent_pos_type is not None ):
			sent_word = ("%s" + format_ + "%s")%(sent_pos_type,w) 
		if(sent_word in sent_scores):
			p,n = sent_scores[sent_word]
		if(w in sent_score_without_ ):
			p_w,n_w = sent_scores_without_[w]
		pos_score.append(p)
		neg_score.append(n)
		pos_score_without.append(p_w)
		neg_score_without.append(n_w)
	avg_pos = np.mean(pos_score)
	avg_neg = np.mean(neg_score)
	avg_pos_without = np.mean(pos_score_without)
	avg_neg_without = np.mean(neg_score_without)
	
	return [1 - avg_pos - avg_neg, avg_pos, avg_neg, 1 - avg_pos_without - avg_neg_without, avg_pos_without, avg_neg_without]

def using_cmu_get_avg_score(data,sent_scores_,sent_scores_without,format_):
	temp_data = data.split(' ')
	sent_word = None
	pos_score = []
	neg_score = [] 
	pos_score_without = []
	neg_score_without = []
	for index,datum in enumerate(temp_data):
		#print(datum)
		pos_str = datum[datum.rfind('_') + 1:].rstrip()
		str_str = datum[:datum.rfind('_')].rstrip()
		sent_pos_type = None
		p,n = 0,0
		p_w,n_w = 0,0
		if(pos_str.lower() == 'a'):
			sent_pos_type = 'a'
		elif(pos_str.lower() == 'n'):
			sent_pos_type = 'n'
		elif(pos_str.lower() == 'd'):
			sent_pos_type = 'r'
		elif(pos_str.lower() == 'v'):
			sent_pos_type = 'v'
		elif(pos_str.lower() == 'p'):
			sent_pos_type = 'p'
		if(sent_pos_type is not None):
			sent_word = ("%s" + format_+ "%s") % (sent_pos_type,str_str.lower())
		if(sent_word in sent_scores_):
			p, n = sent_scores[sent_word]
		if(datum in sent_scores_without):
			print(sent_scores_without[datum])
			p_w,n_w = sent_scores_without[datum]
		pos_score.append(p)
		neg_score.append(n)
		pos_score_without.append(p_w)
		neg_score_without.append(n_w)
	avg_pos_score = np.mean(pos_score)
	avg_neg_score = np.mean(neg_score)
	avg_pos_score_without = np.mean(pos_score_without)
	avg_neg_score_without = np.mean(neg_score_without)

	return [1 - avg_pos_score - avg_neg_score, avg_pos_score,avg_neg_score,1 - avg_pos_score_without - avg_neg_score_without,avg_pos_score_without,avg_neg_score_without ]
def get_polarity_features(source_data,sent_scores,sent_scores_without):
	mat_features = np.zeros((len(source_data),6))
	for index, elem in enumerate(source_data):
		if(len(elem.split('\t')) > 1):
			sub_elem = elem.split('\t')[2]
		else:
		 	sub_elem = elem
		score_list = using_cmu_get_avg_score(sub_elem,sent_scores,sent_scores_without,'_')
		mat_features[index] = score_list
	return mat_features


