#! usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math

class result_collector():
    def __init__(self,P_ = 0,R_ = 0,F1_ = 0):
	self.P = P_
	self.R = R_
	self.F1 = F1_
	self.beta = 0.0
	self.best_combination = None
	return
class fScore(object):
    p = 0
    r = 0
    f1 = {}
    def __init__(self,Tp_ = 0,Fp_ = 0, Tn_ = 0, Fn_ = 0,expecresultList_ = None,precresultList_ = None,test_index_ = None):
        self.Tp = Tp_
        self.Fp = Fp_
        self.Tn = Tn_
        self.Fn = Fn_
        self.expecresultList = expecresultList_
        self.preresultList = precresultList_
	self.test_index = test_index_
        return



def positiveMeasurement(listq,indicator,score):
	file_writer = open('data/avg_shuffle_output/test_indexing.txt','a+')
	for elem in xrange(0,len(listq.expecresultList)):
		if(int(listq.expecresultList[elem]) == int(listq.preresultList[elem])):
			if(int(listq.expecresultList[elem]) == 1):
				score.Tp += 1      
			else:
			 	score.Tn += 1
		else:
		  	if(int(listq.expecresultList[elem]) == 1):
				if(listq.test_index != None):
					file_writer.write(str(listq.test_index[elem])+'\n')
				score.Fn += 1
			else:
			 	score.Fp += 1
	file_writer.close()
	return score
def fscore_measurement(p,r,beta):
    return ((1.0+math.pow(beta,2))*p*r)/((math.pow(beta,2)*p)+r)
def scoreMeasurement(measurementOfscore,beta_val):
    fscoreMeasurement = fScore()
    #fscoreMeasurement = positiveMeasurement(measurementOfscore,1,fscoreMeasurement)
    fscoreMeasurement = positiveMeasurement(measurementOfscore,0,fscoreMeasurement)
    #print(str(fscoreMeasurement.Tp) +' '+ str(fscoreMeasurement.Tn) +' ' + str(fscoreMeasurement.Fn) +' ' + str(fscoreMeasurement.Fp))
    f_score = precisionAndrecall(fscoreMeasurement,beta_val)
    return f_score.f1,f_score.p,f_score.r
def precisionAndrecall(fscoreMeasurement,beta_val):
    #print('Tp: %3.5f'%(fscoreMeasurement.Tp))
    #print('Tn: %3.5f'% (fscoreMeasurement.Tn))
    #print('Fp: %3.5f'%(fscoreMeasurement.Fp))
    #print('Fn: %3.5f '%(fscoreMeasurement.Fn))
    fscoreMeasurement.p = float(fscoreMeasurement.Tp)/(float(fscoreMeasurement.Tp+fscoreMeasurement.Fp))
    fscoreMeasurement.r = float(fscoreMeasurement.Tp)/(float(fscoreMeasurement.Tp+fscoreMeasurement.Fn))
    #print('Precision: %3.2f '% (fscoreMeasurement.p))
    #print('Recall: %3.2f' % (fscoreMeasurement.r))
    fscoreMeasurement.f1 = fscore_measurement(fscoreMeasurement.p,fscoreMeasurement.r,beta_val)
    #print('F-score: %3.5f' %(fscore_measurement(fscoreMeasurement.p,fscoreMeasurement.r)))
    return fscoreMeasurement

